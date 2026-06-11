import json

# 从文件中读取输入数据
with open('output.json', 'r', encoding='utf-8') as input_file:
    data = json.load(input_file)

# 分割文本为句子列表
sentences = data["text"].split("。")
sentences = [s.strip() for s in sentences if s.strip()]

# 计算每个句子的起始位置
sentence_start = [0]
for sentence in sentences[:-1]:
    sentence_start.append(sentence_start[-1] + len(sentence) + 1)

# 创建输出数据列表
output_data = []

# 处理每个句子
for i, sentence in enumerate(sentences):
    # 创建新的数据格式
    new_data = {
        "text": sentence,
        "entities": {}
    }

    # 调整实体位置和内容
    if "entities" in data:
        for key, value in data["entities"].items():
            new_entities = []
            for entity, positions in value:
                new_positions = []
                for pos in positions:
                    if pos >= sentence_start[i] and pos < sentence_start[i] + len(sentence):
                        adjusted_pos = pos - sentence_start[i]
                        new_positions.append(adjusted_pos)  # 调整实体位置
                if new_positions:
                    new_entities.append([entity, new_positions])
            if new_entities:
                new_data["entities"][key] = new_entities

    # 将新数据添加到输出数据列表
    output_data.append(new_data)

# 将输出数据列表写入输出文件
with open('hehe.json', 'w', encoding='utf-8') as output_file:
    for item in output_data:
        json.dump(item, output_file, ensure_ascii=False)
        output_file.write("\n")