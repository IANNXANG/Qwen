import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.stats import gaussian_kde

def plot_distribution(listA: list, name: str):
    # 绘制概率分布图
    data_array = np.array(listA)
    plt.figure(figsize=(10, 6))
    plt.hist(data_array, bins=30, density=True, alpha=0.5, color='b', edgecolor='black')

    # 添加概率密度曲线
    kde = gaussian_kde(data_array)
    x = np.linspace(min(data_array), max(data_array), 1000)
    plt.plot(x, kde(x), color='red', label='Density Curve')

    # 计算均值和标准差
    mean = np.mean(data_array)
    std_dev = np.std(data_array)

    # 绘制均值和标准差
    plt.axvline(mean, color='green', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(mean + std_dev, color='orange', linestyle='dashed', linewidth=1, label='Mean + 1 Std Dev')
    plt.axvline(mean - std_dev, color='orange', linestyle='dashed', linewidth=1, label='Mean - 1 Std Dev')

    # 图形美化
    plt.xlabel('Values', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title(f'Probability Distribution of {name}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 显示图形
    plt.show()

if __name__ == '__main__':

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--work', type=str, default="train", help='The work directory')

    args = parser.parse_args()
    work = args.work

    work = "train"
    # 初始化一个空列表来存储数据
    data_list = []

    # 读取jsonl文件
    with open(f'step_score/{work}_score.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的JSON数据
            data = json.loads(line)
            data_list.append(data)

    listALL = []
    for index, data in enumerate(data_list):
        for key in data:
            listALL.append(data[key])


    # 调用绘图函数
    plot_distribution(listALL,"ALL")
