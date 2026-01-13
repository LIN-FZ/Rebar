import os

folder_path = "D:\\rebar\\data\\valid\\labels"

# 获取文件夹中的所有txt文件
txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

for file in txt_files:
    file_path = os.path.join(folder_path, file)

    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        for line in lines:
            if line.strip() and line[0] == "1":
                line = "0" + line[1:]
            f.write(line)