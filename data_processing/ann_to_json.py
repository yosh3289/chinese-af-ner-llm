import json


# 格式转换
def jsonTransform(jsondata):

    json = {}
    json['text'] = jsondata['text']

    json['label'] = {}
    json['label']["症状体征"] = {}
    json['label']["疾病和诊断"] = {}
    json['label']["治疗方式"] = {}
    json['label']["检验检查"] = {}
    json['label']["身体部位"] = {}

    for ents in jsondata["entities"]:
        # ents  症状体征  疾病和诊断  治疗方式  检验检查  身体部位

        for ent in jsondata["entities"][ents]:

            if ent[0] not in json['label'][ents]:
                json['label'][ents][ent[0]] = [list(ent[1])]
            else:
                json['label'][ents][ent[0]].append(list(ent[1]))

            print(ent)


    print(json)
    return json

def parse_bio_to_json(bio_filename, json_filename):
    # 读取BIO文件

    with open(bio_filename, 'r', encoding='utf-8') as bio_file:
        bio_lines = bio_file.readlines()

    # 初始化变量
    text = ""
    entities = {}
    current_entity = []
    current_type = None
    start_position = None

    # 解析BIO文件
    for line in bio_lines:
        if line.strip():  # 非空行
            parts = line.strip().split()
            word, tag = parts[3], parts[0]

            if tag.startswith('B-'):  # 实体开始
                if current_entity:  # 如果已经在收集实体，则保存
                    entity_text = ''.join(current_entity)
                    entities[current_type].append([entity_text, (start_position, start_position + len(entity_text) - 1)])
                current_entity = [word]  # 开始新实体
                current_type = tag[2:]  # 实体类型
                if current_type not in entities:  # 确保类型键存在
                    entities[current_type] = []
                start_position = len(text)
            elif tag.startswith('I-') and current_entity and tag[2:] == current_type:
                current_entity.append(word)  # 继续收集实体
            else:  # 实体结束或O标签
                if current_entity:  # 如果已经在收集实体，则保存
                    entity_text = ''.join(current_entity)
                    entities[current_type].append([entity_text, (start_position, start_position + len(entity_text) - 1)])
                    current_entity = []  # 重置当前实体
                    current_type = None
            text += word + ''  # 构建文本

    # 处理最后一个实体
    if current_entity:
        entity_text = ''.join(current_entity)
        entities[current_type].append([entity_text, (start_position, start_position + len(entity_text) - 1)])

    print(text)
    print(entities)
    # 构建JSON对象
    bio_json = {
        'text': text.strip(),
        'entities': {etype: [[ent[0], ent[1]] for ent in ents] for etype, ents in entities.items()}
    }
    bio_json = jsonTransform(bio_json)

    print(bio_json)
    # new_data = {
    #     "text": bio_json["text"],
    #     "entities": {
    #         "症状体征": {},
    #         "疾病和诊断": {},
    #         "治疗方式": {},
    #         "检验检查": {},
    #         "身体部位": {}
    #     }
    # }
    #
    # for key, value in bio_json["entities"].items():
    #     print(key)
    #     print(value)
    #     for item in value:
    #         print(item)
    #
    #         new_data["entities"][key][item[0]] = [item[1]]






    # 写入JSON文件
    with open(json_filename, 'w', encoding='utf-8') as json_file: json.dump(bio_json, json_file, ensure_ascii=False, indent=2)
    # 使用示例


for i in range(1, 41):
    instr = 'conlldata/' + str(i) + '.conll'
    outstr = 'jsondata/' + str(i) + '.json'
    parse_bio_to_json(instr, outstr)





