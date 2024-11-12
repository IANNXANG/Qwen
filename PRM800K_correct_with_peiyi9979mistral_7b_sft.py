import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
from PRM_ms import calculate_step_scores

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--work', type=str, default="phase2_test", help='The work directory')


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

def get_scores(input_for_prm):
    return calculate_step_scores(input_for_prm, prm_model, prm_tokenizer, device1, candidate_tokens, step_tag_id)

def GenAndScore(prompt):
    # 设置生成的参数
    num_samples = 20  # 希望生成的不同回答的数量
    temperature = 0.7  # 设置温度，影响随机性
    top_k = 50  # 控制生成单词的范围，top_k 越小，生成的结果越保守

    # 编码输入
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device2)

    # 生成多条回答
    outputs = model.generate(
        inputs,
        max_length=1024,
        num_return_sequences=num_samples,
        do_sample=True,
        temperature=temperature,
        top_k=top_k
    )

    # 解码并输出每条结果
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated text {i + 1}:\n{generated_text}\n")
        scores = get_scores(generated_text)
        print("scores:", scores)

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

all_questions_ratings = extract_ratings_per_question(file_name)
for index, ratings in enumerate(all_questions_ratings):
    print(f"问题 {index + 1}: {ratings}")
    print("第一个-1的位置:", find_first_minus_one_position(ratings))


#PRM初始化
good_token = '+'
bad_token = '-'
step_tag = 'ки'
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "/pubshare/LLM/math-shepherd-mistral-7b-prm"

prm_tokenizer = AutoTokenizer.from_pretrained(model_path)
candidate_tokens = prm_tokenizer.encode(f"{good_token} {bad_token}")[1:]
step_tag_id = prm_tokenizer.encode(f"{step_tag}")[-1]
step_tag_id = 1107

prm_model = AutoModelForCausalLM.from_pretrained(model_path).eval()
prm_model.to(device1)


# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
model = AutoModelForCausalLM.from_pretrained("peiyi9979/mistral-7b-sft").eval()


# 设置模型运行环境
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device2)



# 提取PRM800K的问题和答案作为给prm进行评分
# Bug修复：将zip对象转换为列表
for index, (data, ratings) in enumerate(list(zip(data_list, all_questions_ratings))[:2]):
    print(f"问题 {index + 1}: {ratings}")
    first_minus_one_position = find_first_minus_one_position(ratings)
    print("第一个-1的位置:", first_minus_one_position)
    print(f"Data {index + 1}:")
    problem = data['question']['problem']
    input_for_prm = ""
    count = 0
    for step in data['label']['steps']:
        for completion in step['completions']:
            input_for_prm += completion['text'] + "ки\n"
            count = count + 1
            if count == first_minus_one_position:
                break
        if count == first_minus_one_position:
            break
    input_for_prm = problem + "\n" + input_for_prm
    print(input_for_prm)
    scores = get_scores(input_for_prm)
    print("scores:", scores)





# prompt = "A Senate committee has 5 Democrats, 5 Republicans, and 1 Independent.  In how many ways can they sit around a circular table if all the members of each party all sit next to each other?  (Two seatings are considered equivalent if one is a rotation of the other.)"
#
# #生成并评分
# GenAndScore(prompt)



