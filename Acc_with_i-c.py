import json
import argparse
from collections import OrderedDict
from tabulate import tabulate
from prettytable import PrettyTable


def calculate_accuracy(file_name):
    true_count = 0
    false_count = 0
    # 初始化一个空列表来存储数据
    data_list = []
    # 读取jsonl文件
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的JSON数据
            data = json.loads(line, object_pairs_hook=OrderedDict)
            data_list.append(data)
            if data["score"] == [True]:
                true_count += 1
            elif data["score"] == [False]:
                false_count += 1
    acc = true_count / (true_count + false_count)
    print(f"acc_{path1} = true_count / (total_count):", f"{100 * acc:.2f}% = {true_count} / {true_count + false_count}")
    return acc,true_count,false_count

def get_score_list(file_name):
    score_list = []
    # 初始化一个空列表来存储数据
    data_list = []
    # 读取jsonl文件
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行的JSON数据
            data = json.loads(line, object_pairs_hook=OrderedDict)
            data_list.append(data)
            # 将score值添加到列表中
            score_list.append(data["score"])

    return score_list

def get_IC(path1,path2):
    list1 = get_score_list(path1)
    acc1,_,_ = calculate_accuracy(path1)
    list2 = get_score_list(path2)
    acc2,_,_ = calculate_accuracy(path2)
    C2I = 0
    I2C = 0
    TOTAL = 0
    for i, (item1, item2) in enumerate(zip(list1, list2)):
        if item1 == [True] and item2 == [False]:
            C2I += 1
        elif item1 == [False] and item2 == [True]:
            I2C += 1
        TOTAL += 1

    table = PrettyTable()
    table.field_names = ["Item", "Value"]
    table.add_row(["acc1", f"{100 * acc1:.2f}%"])
    table.add_row(["acc2", f"{100 * acc2:.2f}%"])
    table.add_row(["Δacc", f"{100 * (acc2 - acc1):.2f}%"])
    table.add_row(["C->I", f"{100 * C2I / TOTAL:.2f}%"])
    table.add_row(["I->C", f"{100 * I2C / TOTAL:.2f}%"])
    print(table)


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path1', type=str, default="math/test_pure.jsonl", help='The work directory')
    parser.add_argument('--path2', type=str, default="math/test_cot.jsonl", help='The work directory')

    args = parser.parse_args()
    path1 = args.path1
    path2 = args.path2

    get_IC(path1,path2)