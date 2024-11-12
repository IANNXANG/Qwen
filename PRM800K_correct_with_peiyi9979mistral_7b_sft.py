import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--work', type=str, default="phase2_test", help='The work directory')

args = parser.parse_args()
work = args.work
# PRM800K相关处理
# 初始化一个空列表来存储数据
data_list = []
file_name = f"../PRM800K/{work}.jsonl"
# 读取jsonl文件
with open(file_name, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的JSON数据
        data = json.loads(line)
        data_list.append(data)


# 提取PRM800K的评分保存到一个list中，list中的元素是每一行的评分的一个list
def extract_ratings_per_question(file_path):
    questions_ratings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question_ratings = []
            for step in data['label']['steps']:
                if step['completions'] is not None:  # 专为phase1设计
                    for completion in step['completions']:
                        if completion['rating'] is not None:
                            question_ratings.append(completion['rating'])
                        else:
                            question_ratings.append(None)
                else:
                    input("step['completions'] is None")
            questions_ratings.append(question_ratings)
    return questions_ratings

#找到第一个-1的位置
def find_first_minus_one_position(lst):
    for index, item in enumerate(lst):
        if item == None:
            return None
        elif item == -1:
            return index
    return None

all_questions_ratings = extract_ratings_per_question(file_name)
for index, ratings in enumerate(all_questions_ratings):
    print(f"问题 {index + 1}: {ratings}")
    print("第一个-1的位置:", find_first_minus_one_position(ratings))

# 提取PRM800K的问题和答案
for index, data in enumerate(data_list[:2]):
    print(f"Data {index + 1}:")
    problem = data['question']['problem']
    input_for_prm = ""
    for step in data['label']['steps']:
        for completion in step['completions']:
            input_for_prm += completion['text'] + "ки\n"
    input_for_prm = problem + "\n" + input_for_prm
    print(input_for_prm)

# 提取PRM800K的问题和答案作为给Mistral进行下一步生成的内容
for index, data in enumerate(data_list[:2]):
    print(f"Data {index + 1}:")
    problem = data['question']['problem']
    input_for_prm = ""
    for step in data['label']['steps']:
        for completion in step['completions']:
            input_for_prm += completion['text'] + "ки\n"
    input_for_prm = problem + "\n" + input_for_prm
    print(input_for_prm)

input("继续执行")



# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
model = AutoModelForCausalLM.from_pretrained("peiyi9979/mistral-7b-sft").eval()


# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "A Senate committee has 5 Democrats, 5 Republicans, and 1 Independent.  In how many ways can they sit around a circular table if all the members of each party all sit next to each other?  (Two seatings are considered equivalent if one is a rotation of the other.)"

# 调用模型回答问题
inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
outputs = model.generate(inputs, max_length=1024)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)



