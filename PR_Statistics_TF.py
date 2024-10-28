import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--work', type=str, default="pure", help='The work directory')

args = parser.parse_args()
work = args.work



# 初始化一个空列表来存储数据
data_list = []

# 读取jsonl文件
with open(f'math/eval_{work}_score.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的JSON数据
        data = json.loads(line)
        data_list.append(data)

# 读取jsonl文件
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
    for i, key in enumerate(list(data)[:-1]):
        print(f"Key {i + 1}: {key}")
        print(f"Value {i + 1}: {data[key]}")
        if data["score"] == [True]:
            listTrue.append(data[key])
        elif data["score"] == [False]:
            listFalse.append(data[key])
        print()


        print()

print(listTrue)
print(listFalse)
# 假设listALL是一个包含数值数据的列表
# 这里我们使用随机生成的数据作为示例
np.random.seed(0)

# 绘制概率分布图
plt.hist(listTrue, bins=30, density=True, alpha=0.7, color='b')
plt.xlabel('True alues')
plt.ylabel('Probability Density')
plt.title('Probability Distribution of listALL')
plt.grid(True)
plt.show()

# 绘制概率分布图
plt.hist(listFalse, bins=30, density=True, alpha=0.7, color='b')
plt.xlabel('False Values')
plt.ylabel('Probability Density')
plt.title('Probability Distribution of listALL')
plt.grid(True)
plt.show()