# 定义一个空字符串，用于存储合并后的内容
merged_content = ""

# 循环遍历40个文件
for i in range(1, 41):
    # 构建文件名
    filename = f"resultdata/result{i}.json"

    # 打开文件并读取内容
    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read()

    # 将读取的内容添加到合并后的字符串中，并在每个文件内容之间添加换行符
    merged_content += content + "\n"

# 将合并后的内容写入到一个新文件中
with open("merged_file1.json", "w", encoding="utf-8", errors="ignore") as file:
    file.write(merged_content)