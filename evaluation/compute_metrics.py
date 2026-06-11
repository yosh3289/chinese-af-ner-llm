import json
from typing import List, Dict, Tuple, Any

# 固定五类实体类别
ENTITY_CATEGORIES = ["身体部位", "症状体征", "检验检查", "治疗方式", "疾病和诊断"]

def get_entity_positions(text: str, entities: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """获取实体在原文中的字符起止位置，无匹配返回空列表"""
    pos_dict = {}
    for entity in entities:
        positions = []
        start_idx = 0
        while True:
            idx = text.find(entity, start_idx)
            if idx == -1:
                break
            end_idx = idx + len(entity)
            positions.append((idx, end_idx))
            start_idx = idx + 1
        pos_dict[entity] = positions
    return pos_dict

def calc_tp_fp_fn(gold_pos: Dict[str, List[Tuple[int, int]]],
                  pred_pos: Dict[str, List[Tuple[int, int]]]) -> Dict[str, int]:
    """单类别内部计算TP FP FN，严格按实体+位置完全匹配"""
    tp = 0
    pred_cnt = sum(len(ps) for ps in pred_pos.values())
    gold_cnt = sum(len(ps) for ps in gold_pos.values())

    for ent, pred_ps in pred_pos.items():
        if ent not in gold_pos:
            continue
        for p_pos in pred_ps:
            if p_pos in gold_pos[ent]:
                tp += 1

    fp = pred_cnt - tp
    fn = gold_cnt - tp
    return {"tp": tp, "fp": fp, "fn": fn}

def process_one_sample(sample: Dict) -> Dict[str, Any]:
    """处理单条样本，返回位置与单条分类指标"""
    try:
        # 抽取用户原文、金标准、模型预测
        user_text = next(c["content"] for c in sample["conversations"] if c["role"] == "user")
        gold_raw = next(c["content"] for c in sample["conversations"] if c["role"] == "assistant")
        pred_raw = sample["inference_result"]

        gold_data = json.loads(gold_raw)
        pred_data = json.loads(pred_raw)
    except Exception:
        return {"error": "数据解析失败"}

    # 初始化五类实体
    gold_ents = {cat: gold_data.get(cat, []) for cat in ENTITY_CATEGORIES}
    pred_ents = {cat: pred_data.get(cat, []) for cat in ENTITY_CATEGORIES}

    # 每类实体位置
    gold_pos_all = {cat: get_entity_positions(user_text, gold_ents[cat]) for cat in ENTITY_CATEGORIES}
    pred_pos_all = {cat: get_entity_positions(user_text, pred_ents[cat]) for cat in ENTITY_CATEGORIES}

    # 单条每类指标
    sample_cat_metric = {}
    for cat in ENTITY_CATEGORIES:
        sample_cat_metric[cat] = calc_tp_fp_fn(gold_pos_all[cat], pred_pos_all[cat])

    return {
        **sample,
        "user_original_text": user_text,
        "gold_entity_position": gold_pos_all,
        "pred_entity_position": pred_pos_all,
        "single_sample_category_metric": sample_cat_metric
    }

def aggregate_category_metric(processed_samples: List[Dict]) -> Dict[str, Dict[str, float]]:
    """汇总所有样本，按五类分别计算整体P R F1"""
    total_stat = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in ENTITY_CATEGORIES}

    for samp in processed_samples:
        if "single_sample_category_metric" not in samp:
            continue
        for cat in ENTITY_CATEGORIES:
            sm = samp["single_sample_category_metric"][cat]
            total_stat[cat]["tp"] += sm["tp"]
            total_stat[cat]["fp"] += sm["fp"]
            total_stat[cat]["fn"] += sm["fn"]

    final_res = {}
    for cat in ENTITY_CATEGORIES:
        tp = total_stat[cat]["tp"]
        fp = total_stat[cat]["fp"]
        fn = total_stat[cat]["fn"]

        # 防除0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        final_res[cat] = {
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }
    return final_res

def run_batch_process(input_json_path: str, output_json_path: str):
    """批量入口函数"""
    with open(input_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = [process_one_sample(item) for item in raw_data]
    category_overall = aggregate_category_metric(processed)

    out_data = {
        "all_samples": processed,
        "category_overall_metrics": category_overall,
        "desc": "按身体部位、症状体征、检验检查、治疗方式、疾病和诊断五类分别评估"
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"批量处理完成！结果已保存至：{output_json_path}")

if __name__ == "__main__":
    # 在这里修改你的文件路径
    INPUT_FILE = "input.json"
    OUTPUT_FILE = "output.json"
    run_batch_process(INPUT_FILE, OUTPUT_FILE)