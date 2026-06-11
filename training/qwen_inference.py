import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ====================== 配置项 ======================
BASE_MODEL_DIR = "./Qwen/Qwen3-8B-Instruct"
LORA_MODEL_PATH = "./output/Qwen3_medical_ner_8bit/checkpoint-XXX"  # 替换为你的checkpoint路径
TEST_DATA_PATH = "你的测试集.json"
OUTPUT_RESULT_PATH = "Qwen3_实体识别结果.json"
MAX_NEW_TOKENS = 512
# ===================================================

def extract_entities(messages, model, tokenizer):
    """
    提取医学实体并返回JSON字符串
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
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 提取JSON部分
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            return response[json_start:json_end]
        else:
            return "{}"
    except:
        return "{}"


if __name__ == "__main__":
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model.eval()
    print("模型加载完成！")

    # 加载测试集
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    total = len(test_data)

    for idx, item in enumerate(test_data):
        print(f"正在处理第 {idx+1}/{total} 条样本")
        user_text = next(c["content"] for c in item["conversations"] if c["role"] == "user")
        system_text = next(c["content"] for c in item["conversations"] if c["role"] == "system")

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ]

        start_time = time.time()
        inference_result = extract_entities(messages, model, tokenizer)
        inference_time = round(time.time() - start_time, 6)

        # 生成和你之前完全一致的输出格式
        result_item = {
            **item,
            "inference_result": inference_result,
            "inference_time": inference_time,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
        }
        results.append(result_item)

    # 保存结果
    with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n推理完成！结果已保存至：{OUTPUT_RESULT_PATH}")
    print("现在可以使用你之前的错误分析代码评估模型性能！")