import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
from PRM_ms import calculate_step_scores
from vllm import LLM, SamplingParams


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
    # 设置vllm的采样参数
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_k=top_k,
        max_tokens = 1024
    )

    outputs = llm.generate(prompt, sampling_params)
    sequences = []
    scores = []
    # 解码并输出每条结果
    for index, output in enumerate(outputs):
        for i in range(num_samples):
            generated_text = output.outputs[i].text
            print(f"Generated text {i + 1}:\n{generated_text}\n")
            sequences.append(generated_text)
            score = get_scores(prompt+generated_text)
            print("scores:", score)
            scores.append(score)

    return sequences, scores

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
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_path = "/pubshare/LLM/math-shepherd-mistral-7b-prm"

prm_tokenizer = AutoTokenizer.from_pretrained(model_path)
candidate_tokens = prm_tokenizer.encode(f"{good_token} {bad_token}")[1:]
step_tag_id = prm_tokenizer.encode(f"{step_tag}")[-1]
#step_tag_id2 = 1107

prm_model = AutoModelForCausalLM.from_pretrained(model_path).eval()
prm_model.to(device1)

# 使用vllm的LLM进行生成
llm = LLM(model="peiyi9979/mistral-7b-sft")


# 设置模型运行环境
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")




# 提取PRM800K的问题和答案作为给prm进行评分
# Bug修复：将zip对象转换为列表
for index, (data, ratings) in enumerate(list(zip(data_list, all_questions_ratings))):
    print(f"问题 {index + 1}: {ratings}")
    first_minus_one_position = find_first_minus_one_position(ratings)
    print("第一个-1的位置:", first_minus_one_position)
    print(f"Data {index + 1}:")
    if first_minus_one_position is None:
        continue
    problem = data['question']['problem']
    input_for_prm = ""
    count = 0
    for step in data['label']['steps']:
        for completion in step['completions']:
            input_for_prm += f"step {count+1}: "+completion['text'] + "ки\n"
            count = count + 1
            if count == first_minus_one_position:
                break
        if count == first_minus_one_position:
            break
    input_for_prm = problem + "\n" + input_for_prm
    print(input_for_prm)
    sequences, scores = GenAndScore(input_for_prm)
    values = []
    for score in scores:
        if first_minus_one_position is not None and 0 <= first_minus_one_position < len(score):
            value = float(score[first_minus_one_position])
        else:
            value = -1
        values.append(value)

    print("values:", values)
    # 找到最大的分数
    print("第一个-1的位置", first_minus_one_position)
    print("PRM800K分数:", all_questions_ratings[index])
    max_value = max(values)
    # 找到最大分数的索引
    max_index = values.index(max_value)
    # 打印最大分数和对应的序列
    print("最大分数:", max_value)
    print("对应的序列:", sequences[max_index])
    json_dict = {
        "index": index,
        "第一个-1的位置": None,
        "input_for_prm": None,
        "生成的回答": None,
        "生成的回答的分数": None,
        "生成的回答的分数s": None
    }

    json_dict["第一个-1的位置"] = first_minus_one_position
    json_dict["input_for_prm"] = input_for_prm
    json_dict["生成的回答"] = sequences[max_index]
    json_dict["生成的回答的分数"] = max_value
    json_dict["生成的回答的分数s"] = values
    with open(f'/home/jovyan/notebook/zhouyang/{work}_DPO_DATA.jsonl', 'a', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False)
        f.write('\n')



