import json
import matplotlib.pyplot as plt
import numpy as np





# 初始化一个空列表来存储数据
data_list = []

# 读取jsonl文件
with open('result_score.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的JSON数据
        data = json.loads(line)
        data_list.append(data)


listALL = []
for index, data in enumerate(data_list):
    print(f"Data {index + 1}:")
    print(data)
    for i, key in enumerate(data):
        print(f"Key {i + 1}: {key}")
        print(f"Value {i + 1}: {data[key]}")
        listALL.append(data[key])
        print()

print(listALL)
# 假设listALL是一个包含数值数据的列表
# 这里我们使用随机生成的数据作为示例
np.random.seed(0)

# 绘制概率分布图
plt.hist(listALL, bins=30, density=True, alpha=0.7, color='b')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.title('Probability Distribution of listALL')
plt.grid(True)
plt.show()