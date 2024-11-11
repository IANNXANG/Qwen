import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PR_Statistics import plot_distribution

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--work', type=str, default="phase2_test", help='The work directory')

args = parser.parse_args()
work = args.work


def extract_ratings_per_question(file_path):
    questions_ratings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question_ratings = []
            for step in data['label']['steps']:
                if step['completions'] is not None:  #专为phase1设计
                    for completion in step['completions']:
                        if completion['rating'] is not None:
                            question_ratings.append(completion['rating'])
                        else:
                            question_ratings.append(None)
                else:
                    input("step['completions'] is None")
            questions_ratings.append(question_ratings)
    return questions_ratings

# 初始化一个空列表来存储数据
data_list = []

# 读取jsonl文件score
with open(f'PRM800K/{work}_score.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的JSON数据
        data = json.loads(line)
        data_list.append(data)

print(data_list)

# 假设数据文件路径为 'data/prm800k_data.jsonl'
data_list_PRM800K = extract_ratings_per_question(f'../PRM800K/{work}.jsonl')
print(data_list_PRM800K)




listTrue = []
list0 = []
listFalse = []
listNone = []
# 假设listALL是一个包含数值数据的列表
for index, (data, data_PRM800K) in enumerate(zip(data_list, data_list_PRM800K)):
    print(f"Data {index + 1}:")
    print(data)
    print(data_PRM800K)
    for i, (score, label) in enumerate(zip(data, data_PRM800K)):
        print(f"data {i + 1}: {data[f"step{i+1}_score"]}")
        print(f"data_PRM800K {i + 1}: {label}")
        if label == 1:
            listTrue.append(data[f"step{i+1}_score"])
        elif label == 0:
            list0.append(data[f"step{i+1}_score"])
        elif label == -1:
            listFalse.append(data[f"step{i+1}_score"])
        elif label == None:
            listNone.append(data[f"step{i+1}_score"])
        print()

        print()


plot_distribution(listTrue, "1")
plot_distribution(list0, "0")
plot_distribution(listFalse, "-1")
plot_distribution(listNone, "None")

