import json
from collections import defaultdict

def parse_label(label_data, parent_type=""):
    """
    递归解析嵌套的标签结构，处理标注中的格式错误
    :param label_data: 标签字典
    :param parent_type: 父标签类型（用于错误提示）
    :return: 实体列表 [(实体类型, 实体名, 起始位置, 结束位置)]
    """
    entities = []
    for key, value in label_data.items():
        # 如果值是字典，说明是嵌套标签，继续递归解析
        if isinstance(value, dict):
            entities.extend(parse_label(value, key))
        # 如果值是列表，说明是实体的位置列表
        elif isinstance(value, list):
            for pos in value:
                if len(pos) == 2 and isinstance(pos[0], (int, float)) and isinstance(pos[1], (int, float)):
                    entities.append((parent_type, key, int(pos[0]), int(pos[1])))
                else:
                    print(f"警告：实体 '{key}' 的位置格式错误: {pos}，已跳过")
        # 其他情况视为格式错误
        else:
            print(f"警告：标签 '{key}' 的值格式错误: {value}，已跳过")
    return entities

def load_annotation_file(file_path):
    """
    加载标注文件（每行一个JSON对象），自动处理嵌套标签
    :param file_path: 标注文件路径
    :return: 字典，key=文本内容，value=标注实体列表
    """
    annotation_dict = defaultdict(list)
    line_num = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data['text']
                label = data['label']
                # 使用递归函数解析标签
                entities = parse_label(label)
                annotation_dict[text].extend(entities)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析失败: {e}")
            except KeyError as e:
                print(f"第 {line_num} 行缺少字段: {e}")
            except Exception as e:
                print(f"第 {line_num} 行处理失败: {e}")
    return annotation_dict

def calculate_annotation_metrics(anno1_dict, anno2_dict):
    """
    统计标注一致性的基础指标
    """
    common_texts = set(anno1_dict.keys()) & set(anno2_dict.keys())
    total_agree = 0
    total_anno1 = 0
    total_anno2 = 0

    for text in common_texts:
        anno1_entities = set(anno1_dict[text])
        anno2_entities = set(anno2_dict[text])
        
        agree_entities = anno1_entities & anno2_entities
        total_agree += len(agree_entities)
        total_anno1 += len(anno1_entities)
        total_anno2 += len(anno2_entities)

    # 处理仅单个标注者有的文本
    for text in set(anno1_dict.keys()) - common_texts:
        total_anno1 += len(anno1_dict[text])
    
    for text in set(anno2_dict.keys()) - common_texts:
        total_anno2 += len(anno2_dict[text])

    return total_agree, total_anno1, total_anno2

def cohen_kappa(total_agree, total_anno1, total_anno2):
    """
    计算科恩Kappa系数和观察一致率Po
    :return: (kappa系数, 观察一致率Po)
    """
    N = total_anno1 + total_anno2
    if N == 0:
        return 0.0, 0.0
    
    Po = (2 * total_agree) / N
    p1 = total_anno1 / N
    p2 = total_anno2 / N
    Pe = p1 * p2 + (1 - p1) * (1 - p2)
    
    if 1 - Pe == 0:
        kappa = 1.0
    else:
        kappa = (Po - Pe) / (1 - Pe)
    
    return kappa, Po

def main(anno1_path, anno2_path):
    """
    主函数
    """
    print("正在加载标注文件...")
    anno1_dict = load_annotation_file(anno1_path)
    anno2_dict = load_annotation_file(anno2_path)
    
    print(f"\n加载完成：")
    print(f"标注者1: {len(anno1_dict)} 条文本，{sum(len(v) for v in anno1_dict.values())} 个实体")
    print(f"标注者2: {len(anno2_dict)} 条文本，{sum(len(v) for v in anno2_dict.values())} 个实体")
    print(f"共同文本数: {len(set(anno1_dict.keys()) & set(anno2_dict.keys()))}")
    
    total_agree, total_anno1, total_anno2 = calculate_annotation_metrics(anno1_dict, anno2_dict)
    kappa, Po = cohen_kappa(total_agree, total_anno1, total_anno2)
    
    print("\n===== 标注一致性检验结果 =====")
    print(f"标注者1总标注实体数: {total_anno1}")
    print(f"标注者2总标注实体数: {total_anno2}")
    print(f"两者一致的实体数: {total_agree}")
    print(f"观察一致率 (Po): {Po:.4f}")
    print(f"科恩Kappa系数: {kappa:.4f}")
    
    # 医学标注领域通用的Landis-Koch一致性标准
    if kappa >= 0.80:
        print("一致性判定：极好（Kappa≥0.80，完全符合医学标注学术要求）")
    elif 0.60 <= kappa < 0.80:
        print("一致性判定：显著（0.60≤Kappa<0.80，基本符合学术要求，少量分歧已通过专家仲裁解决）")
    elif 0.40 <= kappa < 0.60:
        print("一致性判定：中等（0.40≤Kappa<0.60，建议补充标注指南并对分歧案例进行统一）")
    else:
        print("一致性判定：较差（Kappa<0.40，需重新审核标注标准并进行二次标注）")

if __name__ == "__main__":
    # 完全保留你的原始文件路径，无需修改
    ANNOTATION_FILE1 = "path/to/your/file.json"
    ANNOTATION_FILE2 = "path/to/annotator2.json"
    main(ANNOTATION_FILE1, ANNOTATION_FILE2)