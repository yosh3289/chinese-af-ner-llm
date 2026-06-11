# 第三种方法
import re


def generate_ann_annotations(data, entity_mapping):
    annotations = []
    entity_counts = 1
    for entity_type, entity_phrases in entity_mapping.items():
        for entity_phrase in entity_phrases:
            # 使用正则表达式匹配识别的词组
            matches = re.finditer(entity_phrase, data)

            for match in matches:
                start = match.start()
                end = match.end()
                entity_id = f'T{entity_counts}'
                annotation = f'{entity_id}\t{entity_type} {start} {end}\t{data[start:end]}'
                annotations.append(annotation)
                entity_counts += 1



 # 按照start标引数字从小到大排序标注结果
    annotations.sort(key=lambda x: int(x.split('\t')[1].split(' ')[1]))

    # 逐个增加实体ID
    new_annotations = []
    entity_id_counter = 1
    for annotation in annotations:
        parts = annotation.split('\t')
        entity_id = parts[0]
        entity_type = parts[1].split(' ')[0]
        start = parts[1].split(' ')[1]
        end = parts[1].split(' ')[2]
        text = parts[2]
        new_entity_id = f'T{entity_id_counter}'
        entity_id_counter += 1
        new_annotation = f'{new_entity_id}\t{entity_type} {start} {end}\t{text}'
        new_annotations.append(new_annotation)

    return new_annotations

def convert_to_ann_format(file_path, entity_mapping):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    # 修正换行符导致的索引偏移
    data = re.sub(r'\n+', ' ', data)
    annotations = generate_ann_annotations(data, entity_mapping)
    return annotations


file_path = "1-50.txt"  # 修改为实际的文件路径

entity_mapping = {
    "身体部位": ["心尖区","二尖瓣听诊区","口腔","生殖器","胸腔","两侧胸廓","胸廓","头颅","五官","双瞳孔","外耳道","鼻翼","双扁桃体","颈","气管","腹部","甲状腺","胸部","肠","肛门","外生殖器","脊柱","四肢","腹壁","跟腱","双侧膝","口唇指趾端","双侧胸廓","双侧语颤","胸膜","心前区","心脏瓣膜","P2","双下肢","左心室","颈静脉","双肾","眼睑","左第5肋间锁骨","心包","心脏","肺部","双肺"],
    "症状体征": ["晕厥","畏寒","干湿啰音","哮鸣音","心慌","溃疡","胸痛","肿大","扇动","反复胸闷","活动后胸闷","气促","反复活动后胸闷","心悸","乏力","外伤","痛经","发绀","杵状指","怒张","清音","干湿罗音","隆起","凹陷","杂音","震颤","不亢","亢进","水肿","感冒","发热","头痛","头晕","腹痛","腹胀","形态正常","充血","倒锥形改变","囊肿","干、湿罗音","呼吸困难","眩晕","扩大","积液","咯血","咳嗽","流脓","畸形","咳痰","呼吸音粗"],
    "检验检查": ["胃镜","心肺五项","身高","收缩压","查体","T","R","体温","脉搏","BP","叩诊","心率","心尖搏动","体重","抽血检查","心电图","三大常规","凝血功能","冠脉造影","心血管造影","肝肾功能","肝、肾功能","超声心动图","CT","MRI" ,"彩超","电解质","血型","胸片"],
    "治疗方式": ["监测血氧饱和度","心电监护","护心","降压","补液","镇痛治疗","控制血压","吸氧","强心","利尿","二尖瓣成形术","手术","输血","一级护理","二级护理"],
    "疾病和诊断": ["室性早搏","PH","心律不齐","乙肝","心律失常","房颤","肝炎","结核","疟疾","高血压","心脏病","糖尿病","脑血管疾病","精神疾病","瓣膜性心脏病","风湿性二尖瓣狭窄","风湿性二尖瓣返流","风湿性主动脉瓣返流","风湿性三尖瓣返流","冠心病","先心病","传染病","风湿性心脏病","风湿性二尖瓣狭窄伴关闭不全","心房纤颤","先天性心脏病","心脏瓣膜病","二尖瓣返流","房间隔缺损","阵发性房颤","冠状动脉粥样硬化性心脏病","腔隙性脑梗","脑梗","慢性胃炎","牙髓炎","前列腺增生","关节炎","青光眼","胆囊结石","胆囊炎","左心房粘液瘤","二尖瓣后叶脱垂","风湿性主动脉瓣返流","肺动脉高压","肥厚性梗阻型心肌病","肺结节","心房颤动","主动脉瓣中度关闭不全","二尖瓣狭窄","三尖瓣返流","慢性支气管炎","MR","ASD","TR","AR","Af","AF","风心病"],

}

annotations = convert_to_ann_format(file_path, entity_mapping)

# 保存标注内容到.ann文件
with open("output.ann", 'w', encoding='utf-8') as file:
    file.write('\n'.join(annotations))

