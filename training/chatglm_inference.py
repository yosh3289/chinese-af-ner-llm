# -*- coding: utf-8 -*-
"""
ChatGLM3-6B (P-Tuning v2) 推理脚本（中文房颤电子病历 NER）。

说明 / NOTE
-----------
与本仓库 chatglm_ptuning.py 配套：加载基座 ChatGLM3-6B 并注入训练得到的
P-Tuning v2 前缀权重，对测试集逐条抽取实体并输出 JSON。输出格式与 qwen_inference.py
完全一致（inference_result / inference_time / processed_at），便于用同一套评测脚本对比。
"""
import json
import os
import time

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ====================== 配置 ======================
BASE_MODEL_DIR = "THUDM/chatglm3-6b"
PTUNING_CHECKPOINT = "./output/ChatGLM3_ptuning_ner/checkpoint-XXXX"  # 替换为你的 checkpoint 路径
TEST_DATA_PATH = "你的测试集.json"
OUTPUT_RESULT_PATH = "ChatGLM3_实体识别结果.json"
PRE_SEQ_LEN = 128
MAX_NEW_TOKENS = 512
QUANTIZATION_BIT = 4   # 与训练保持一致；设为 None 可关闭
# =================================================


def extract_json(response: str) -> str:
    """从模型输出中截取 JSON 片段，失败返回 '{}'。"""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            return response[start:end]
        return "{}"
    except Exception:
        return "{}"


if __name__ == "__main__":
    print("正在加载 ChatGLM3-6B 与 P-Tuning v2 前缀权重 ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)

    # 启用 P-Tuning v2 结构
    config = AutoConfig.from_pretrained(
        BASE_MODEL_DIR, trust_remote_code=True,
        pre_seq_len=PRE_SEQ_LEN, prefix_projection=False,
    )
    model = AutoModel.from_pretrained(BASE_MODEL_DIR, config=config, trust_remote_code=True)

    # 注入训练得到的前缀权重（ChatGLM 官方加载方式）
    prefix_state_dict = torch.load(
        os.path.join(PTUNING_CHECKPOINT, "pytorch_model.bin"), map_location="cpu"
    )
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    if QUANTIZATION_BIT is not None:
        model = model.quantize(QUANTIZATION_BIT)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model.eval()
    print("模型加载完成！")

    # 载入测试集
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    total = len(test_data)
    for idx, item in enumerate(test_data):
        print(f"正在处理 {idx + 1}/{total}")
        user_text = next(c["content"] for c in item["conversations"] if c["role"] == "user")
        system_text = next(c["content"] for c in item["conversations"] if c["role"] == "system")

        start_time = time.time()
        with torch.no_grad():
            response, _ = model.chat(
                tokenizer,
                user_text,
                history=[{"role": "system", "content": system_text}],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
            )
        inference_time = round(time.time() - start_time, 6)

        result_item = {
            **item,
            "inference_result": extract_json(response),
            "inference_time": inference_time,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results.append(result_item)

    with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n推理完成，结果已保存至 {OUTPUT_RESULT_PATH}")
    print("可直接用 evaluation/compute_metrics.py 计算 P/R/F1。")
