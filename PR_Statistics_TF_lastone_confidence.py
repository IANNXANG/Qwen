import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


# 定义绘图函数
def plot_correct_rate(midpoints, correct_rates):
    plt.figure(figsize=(10, 6))

    # 绘制正确率折线图
    plt.plot(midpoints, correct_rates, marker='o', color='blue', label='正确率', linewidth=2)

    # 添加y=x基准线
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='y=x 基准线')

    # 图形美化
    plt.xlabel('奖励值区间中点', fontsize=14)
    plt.ylabel('正确率', fontsize=14)
    plt.title('奖励值区间正确率折线图', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.axhline(0.5, color='grey', linestyle=':', linewidth=1)  # 50%参考线
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 显示图形
    plt.show()

if __name__ == "__main__":
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
        for i, line in enumerate(file):
            # 解析每一行的JSON数据
            data = json.loads(line)
            data_list[i]["score"] = data["score"]

    listTrue = []
    listFalse = []
    for index, data in enumerate(data_list):
        num = len(data) - 1
        if data["score"] == [True]:
            listTrue.append(data[f"step{num}_score"])
        elif data["score"] == [False]:
            listFalse.append(data[f"step{num}_score"])

    # 定义奖励区间
    intervals = [(i / 20, (i + 1) / 20) for i in range(0, 20)]

    # 统计每个奖励区间正确和错误的数量
    correct_counts = [0] * len(intervals)
    wrong_counts = [0] * len(intervals)

    for value in listTrue:
        for i, interval in enumerate(intervals):
            if interval[0] <= value <= interval[1]:
                correct_counts[i] += 1
                break

    for value in listFalse:
        for i, interval in enumerate(intervals):
            if interval[0] <= value <= interval[1]:
                wrong_counts[i] += 1
                break

    # 计算每个奖励区间的正确率
    correct_rates = []
    for correct_count, wrong_count in zip(correct_counts, wrong_counts):
        total = correct_count + wrong_count
        if total == 0:
            correct_rates.append(0)
        else:
            correct_rates.append(correct_count / total)

    # 提取奖励区间的中点作为横坐标
    midpoints = [(interval[0] + interval[1]) / 2 for interval in intervals]

    # 调用绘图函数
    plot_correct_rate(midpoints, correct_rates)
