import json
import time
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
import os
import swanlab

# ====================== 24GB显存专属配置 ======================
MODEL_NAME = "Qwen/Qwen3-8B-Instruct"
TRAIN_DATA_PATH = "你的训练集.json"  # 
TEST_DATA_PATH = "你的测试集.json"
OUTPUT_DIR = "./output/Qwen3_medical_ner_8bit"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 3
LEARNING_RATE = 1.5e-4
# ===========================================================

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将你的原始医学实体数据集转换为大模型微调所需格式
    完全兼容你的conversations数组格式
    """
    messages = []

    with open(origin_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)
        for item in raw_data:
            conversations = item["conversations"]
            # 提取system、user、assistant内容
            system_content = next(c["content"] for c in conversations if c["role"] == "system")
            user_content = next(c["content"] for c in conversations if c["role"] == "user")
            assistant_content = next(c["content"] for c in conversations if c["role"] == "assistant")
            
            message = {
                "instruction": system_content,
                "input": user_content,
                "output": assistant_content,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    数据集预处理，适配Qwen3对话模板
    """
    input_ids, attention_mask, labels = [], [], []
    
    # Qwen3标准对话模板
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    # 拼接输入和输出
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 只计算assistant部分的loss，system和user部分设为-100
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断超长序列
    if len(input_ids) > MAX_SEQ_LENGTH:
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:MAX_SEQ_LENGTH]
        labels = labels[:MAX_SEQ_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    """
    单条样本预测函数
    """
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


if __name__ == "__main__":
    # 1. 下载并加载Qwen3模型和分词器
    print("正在下载Qwen3模型...")
    model_dir = snapshot_download(MODEL_NAME, cache_dir="./", revision="master")
    
    print("正在加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 避免训练时警告

    # 8bit量化配置（24GB显存完美适配）
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()  # 开启梯度检查点必需

    # 2. LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=32,  # 8bit下增大秩提升拟合能力
        lora_alpha=64,
        lora_dropout=0.05,
    )

    model = get_peft_model(model, lora_config)
    print("可训练参数占比：")
    model.print_trainable_parameters()

    # 3. 处理数据集
    train_jsonl_new_path = "new_train.jsonl"
    test_jsonl_new_path = "new_test.jsonl"

    if not os.path.exists(train_jsonl_new_path):
        dataset_jsonl_transfer(TRAIN_DATA_PATH, train_jsonl_new_path)
    if not os.path.exists(test_jsonl_new_path):
        dataset_jsonl_transfer(TEST_DATA_PATH, test_jsonl_new_path)

    # 加载训练集
    train_df = pd.read_json(train_jsonl_new_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

    # 4. 训练参数配置
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=10,
        num_train_epochs=EPOCHS,
        save_steps=100,
        learning_rate=LEARNING_RATE,
        save_on_each_node=True,
        gradient_checkpointing=True,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        save_total_limit=3,
    )

    # 5. SwanLab训练可视化配置
    swanlab_callback = SwanLabCallback(
        project="Qwen3_Medical_NER",
        experiment_name="Qwen3-8B-Instruct_8bit_LoRA",
        description="使用Qwen3-8B-Instruct模型在房颤电子病历实体识别数据集上微调",
        config={
            "model": MODEL_NAME,
            "dataset": "房颤电子病历实体数据集",
            "quantization": "8bit",
            "lora_r": 32,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH
        }
    )

    # 6. 初始化训练器并开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    print("开始训练...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"训练完成，LoRA权重已保存至：{OUTPUT_DIR}")

    # 7. 测试集前10条样本预测
    print("\n正在测试模型效果...")
    test_df = pd.read_json(test_jsonl_new_path, lines=True)[:10]
    test_text_list = []

    for index, row in test_df.iterrows():
        instruction = row['instruction']
        input_value = row['input']
        ground_truth = row['output']

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        start_time = time.time()
        response = predict(messages, model, tokenizer)
        inference_time = round(time.time() - start_time, 4)

        messages.append({"role": "assistant", "content": f"{response}"})
        result_text = f"【输入】\n{input_value}\n\n【标准答案】\n{ground_truth}\n\n【模型输出】\n{response}\n\n【推理时间】{inference_time}s"
        test_text_list.append(swanlab.Text(result_text, caption=f"样本{index+1}"))

    swanlab.log({"Prediction_Results": test_text_list})
    swanlab.finish()