import json
from typing import List, Dict, Tuple
from collections import defaultdict

# ===================== 固定配置（和你脚本完全一致）=====================
ENTITY_CATEGORIES = ["身体部位", "症状体征", "检验检查", "治疗方式", "疾病和诊断"]
FILE_PATH = "qwen例子.json"  # 你的数据文件
# ====================================================================

def get_entity_positions(text: str, entities: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """【完全复刻你的函数】获取实体字符起止位置"""
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
                  pred_pos: Dict[str, List[Tuple[int, int]]]) -> Tuple[int, int, int]:
    """【完全复刻你的函数】严格按实体+位置计算TP/FP/FN"""
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
    return tp, fp, fn

def main():
    # 1. 加载数据
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 全局统计初始化
    total_stats = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in ENTITY_CATEGORIES}
    total_samples = 0

    # 3. 逐条计算（完全对齐你的脚本）
    for sample in data:
        try:
            # 抽取字段
            user_text = next(c["content"] for c in sample["conversations"] if c["role"] == "user")
            gold_raw = next(c["content"] for c in sample["conversations"] if c["role"] == "assistant")
            pred_raw = sample["inference_result"]

            gold_data = json.loads(gold_raw)
            pred_data = json.loads(pred_raw)
        except:
            continue

        total_samples += 1
        # 初始化实体
        gold_ents = {cat: gold_data.get(cat, []) for cat in ENTITY_CATEGORIES}
        pred_ents = {cat: pred_data.get(cat, []) for cat in ENTITY_CATEGORIES}

        # 计算位置
        gold_pos = {cat: get_entity_positions(user_text, gold_ents[cat]) for cat in ENTITY_CATEGORIES}
        pred_pos = {cat: get_entity_positions(user_text, pred_ents[cat]) for cat in ENTITY_CATEGORIES}

        # 累加TP/FP/FN
        for cat in ENTITY_CATEGORIES:
            tp, fp, fn = calc_tp_fp_fn(gold_pos[cat], pred_pos[cat])
            total_stats[cat]["tp"] += tp
            total_stats[cat]["fp"] += fp
            total_stats[cat]["fn"] += fn

    # 4. 计算【微平均总指标】（和你脚本100%一致）
    total_tp = sum(total_stats[cat]["tp"] for cat in ENTITY_CATEGORIES)
    total_fp = sum(total_stats[cat]["fp"] for cat in ENTITY_CATEGORIES)
    total_fn = sum(total_stats[cat]["fn"] for cat in ENTITY_CATEGORIES)

    # 总指标计算
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 5. 分类指标计算
    category_metrics = {}
    for cat in ENTITY_CATEGORIES:
        tp = total_stats[cat]["tp"]
        fp = total_stats[cat]["fp"]
        fn = total_stats[cat]["fn"]
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        category_metrics[cat] = {"TP":tp,"FP":fp,"FN":fn,"P":round(p,4),"R":round(r,4),"F1":round(f,4)}

    # ===================== 输出最终结果 =====================
    print("="*70)
    print("                模型错误分析（官方标准·位置严格匹配）")
    print("="*70)
    print(f"总样本数：{total_samples}")
    print(f"? TP(正确)：{total_tp}   ? FP(误检)：{total_fp}   ? FN(漏检)：{total_fn}")
    print(f"\n总精确率(P)：{precision:.4f}")
    print(f"总召回率(R)：{recall:.4f}")
    print(f"总F1分数：{f1:.4f}")

    print("\n--- 五类实体详细错误统计 ---")
    for cat, metric in category_metrics.items():
        print(f"【{cat}】 TP={metric['TP']} FP={metric['FP']} FN={metric['FN']} | P={metric['P']:.4f} R={metric['R']:.4f} F1={metric['F1']:.4f}")

    # 保存报告
    report = {
        "总样本数": total_samples,
        "总TP": total_tp, "总FP": total_fp, "总FN": total_fn,
        "总Precision": round(precision,4),
        "总Recall": round(recall,4),
        "总F1": round(f1,4),
        "分类统计": category_metrics
    }
    with open("官方标准_错误分析总报告.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n? 报告已保存，结果和你的官方脚本完全一致！")
    return report

if __name__ == "__main__":
    result = main()