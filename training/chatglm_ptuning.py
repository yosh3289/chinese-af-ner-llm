# -*- coding: utf-8 -*-
"""
ChatGLM3-6B + P-Tuning v2 微调脚本（中文房颤电子病历 NER）。

说明 / NOTE
-----------
复现论文 Table 3 的超参数配置（P-Tuning v2, lr=2e-2, 等效 batch=16,
序列长度 128, 3000 steps），并采用与本仓库 Qwen3 脚本完全一致的数据格式与任务设定
（conversations 中 system/user/assistant，assistant 内容为金标实体 JSON）。

遵循 ChatGLM3-6B 官方的 P-Tuning v2 方式（通过 config.pre_seq_len 启用
PrefixEncoder，仅训练前缀参数，冻结主干）。首次运行请按你的环境核对路径、
显存与依赖版本。
"""
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ====================== 配置（对应论文 Table 3） ======================
MODEL_NAME = "THUDM/chatglm3-6b"
TRAIN_DATA_PATH = "你的训练集.json"
TEST_DATA_PATH = "你的测试集.json"
OUTPUT_DIR = "./output/ChatGLM3_ptuning_ner"

PRE_SEQ_LEN = 128                 # P-Tuning v2 前缀长度（即论文 sequence_length 128）
MAX_SOURCE_LEN = 128              # system+user 最大长度
MAX_TARGET_LEN = 256             # assistant（实体 JSON）最大长度
LEARNING_RATE = 2e-2             # P-Tuning v2 标准学习率（与 LoRA 的 1.5e-4 不同，属正常）
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # 等效 batch size = 16
MAX_STEPS = 3000
QUANTIZATION_BIT = 4             # 4-bit 量化以适配消费级显存；设为 None 可关闭
# 五类实体（与标注、评测脚本保持一致）：身体部位 / 症状表现 / 检查检验 / 治疗方式 / 疾病及诊断
# ====================================================================


def dataset_jsonl_transfer(origin_path, new_path):
    """把原始 conversations 数据转成 instruction/input/output 三段式（与 Qwen 脚本一致）。"""
    messages = []
    with open(origin_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        for item in raw_data:
            conv = item["conversations"]
            system_content = next(c["content"] for c in conv if c["role"] == "system")
            user_content = next(c["content"] for c in conv if c["role"] == "user")
            assistant_content = next(c["content"] for c in conv if c["role"] == "assistant")
            messages.append({
                "instruction": system_content,
                "input": user_content,
                "output": assistant_content,
            })
    with open(new_path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def process_func(example):
    """按 ChatGLM3 对话格式构造监督样本：仅对 assistant 段计算 loss。"""
    # 用 ChatGLM3 自带的对话模板构造 prompt（system 作为历史，user 作为当前轮）
    prompt_ids = tokenizer.build_chat_input(
        example["input"],
        history=[{"role": "system", "content": example["instruction"]}],
        role="user",
    )["input_ids"][0].tolist()

    answer_ids = tokenizer.encode(example["output"], add_special_tokens=False)
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    # prompt 部分置 -100，只学 assistant 的输出
    labels = [-100] * len(prompt_ids) + answer_ids

    max_len = MAX_SOURCE_LEN + MAX_TARGET_LEN
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    return {"input_ids": input_ids, "labels": labels}


if __name__ == "__main__":
    print("正在加载 ChatGLM3-6B 与分词器 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 关键：通过 config.pre_seq_len 启用 P-Tuning v2（PrefixEncoder）
    config = AutoConfig.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        pre_seq_len=PRE_SEQ_LEN, prefix_projection=False,
    )
    model = AutoModel.from_pretrained(MODEL_NAME, config=config, trust_remote_code=True)

    if QUANTIZATION_BIT is not None:
        print(f"使用 {QUANTIZATION_BIT}-bit 量化")
        model = model.quantize(QUANTIZATION_BIT)
    model = model.half().cuda()
    # PrefixEncoder 保持 fp32，保证训练稳定（ChatGLM 官方做法）
    model.transformer.prefix_encoder.float()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 冻结主干，只训练前缀参数
    for name, param in model.named_parameters():
        if "prefix_encoder" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    # 数据集
    train_jsonl = "new_train_glm.jsonl"
    if not os.path.exists(train_jsonl):
        dataset_jsonl_transfer(TRAIN_DATA_PATH, train_jsonl)
    import pandas as pd
    train_df = pd.read_json(train_jsonl, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

    # 训练参数（对齐论文 Table 3）
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
        ),
    )

    print("开始训练 ...")
    trainer.train()
    # 仅保存 P-Tuning v2 前缀权重
    trainer.save_model(OUTPUT_DIR)
    print(f"训练完成，前缀权重已保存至 {OUTPUT_DIR}")
