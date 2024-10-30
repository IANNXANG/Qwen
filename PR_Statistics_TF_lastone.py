import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PR_Statistics import plot_distribution

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--work', type=str, default="train", help='The work directory')

args = parser.parse_args()
work = args.work

work = "solution"


# 初始化一个空列表来存储数据
data_list = []

# 读取jsonl文件score
with open(f'step_score/eval_{work}_score.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的JSON数据
        data = json.loads(line)
        data_list.append(data)

# 读取jsonl文件true false
with open(f'math/test_{work}.jsonl', 'r', encoding='utf-8') as file:
    for i ,line in enumerate(file):
        # 解析每一行的JSON数据
        data = json.loads(line)
        data_list[i]["score"] = data["score"]


listTrue = []
listFalse = []
for index, data in enumerate(data_list):
    print(f"Data {index + 1}:")
    print(data)
    num = len(data) - 1
    if data["score"] == [True]:
        listTrue.append(data[f"step{num}_score"])
    elif data["score"] == [False]:
        listFalse.append(data[f"step{num}_score"])


print(listTrue)
print(listFalse)
# 假设listALL是一个包含数值数据的列表


plot_distribution(listTrue, "True")
plot_distribution(listFalse, "False")
