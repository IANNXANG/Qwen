import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
from PRM_ms import calculate_step_scores
import pprint



#prm返回分数 保存为一个list
def get_scores(input_for_prm):
    return calculate_step_scores(input_for_prm, prm_model, prm_tokenizer, device1, candidate_tokens, step_tag_id)

def GenAndScore(prompt):
    # 设置生成的参数
    num_samples = 1  # 希望生成的不同回答的数量
    temperature = 0.7  # 设置温度，影响随机性
    top_k = 50  # 控制生成单词的范围，top_k 越小，生成的结果越保守

    # 编码输入
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device2)

    # 生成多条回答
    outputs = model.generate(
        inputs,
        max_new_tokens=1024,
        num_return_sequences=num_samples,
        do_sample=True,
        temperature=temperature,
        top_k=top_k
    )
    sequences = []
    scores = []
    # 解码并输出每条结果
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated text {i + 1}:\n{generated_text}\n")
        sequences.append(generated_text)
        score = get_scores(generated_text)
        print("scores:", score)
        scores.append(score)

    return sequences, scores


# train.jsonl相关处理
# 读取 JSONL 文件
with open('math/train.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]





#PRM初始化放置在GPU0上
good_token = '+'
bad_token = '-'
step_tag = 'ки'
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "/pubshare/LLM/math-shepherd-mistral-7b-prm"

prm_tokenizer = AutoTokenizer.from_pretrained(model_path)
candidate_tokens = prm_tokenizer.encode(f"{good_token} {bad_token}")[1:]
step_tag_id = prm_tokenizer.encode(f"{step_tag}")[-1]
#step_tag_id2 = 1107

prm_model = AutoModelForCausalLM.from_pretrained(model_path).eval()
prm_model.to(device1)


#Actor模型放置在GPU1上
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct").eval()
# 设置模型运行环境
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device2)


# 打印读取到的 JSON 数据
for item in data:
    print("------------------------------------------------------------------------------------")
    print(f"问题：{item['problem']}\n答案：{item['answer']}")
    inputs = tokenizer(item['problem'] + "\n\n", return_tensors="pt").to(device2)
    outputs = model.generate(**inputs, max_length=2048)
    # print("outputs[0]：\n",outputs[0])
    # print("outputs：\n",outputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    parts = answer.split("\n\n")
    result_dict = {}
    for index, part in enumerate(parts):
        key = f"step{index}" if index > 0 else "question"
        result_dict[key] = part
    pprint.pprint(result_dict)

    #创建input_for_prm第一次
    input_for_prm = ""
    for i, key in enumerate(result_dict):
        print(f"{key}: {result_dict[key]}")
        if i == 0:
            input_for_prm = input_for_prm + result_dict[key]
        else:
            input_for_prm = input_for_prm + result_dict[key] + 'ки'
    input_for_prm = input_for_prm.replace("<|im_end|>", "")

    #使用input_for_prm生成分数
    scores1 = get_scores(input_for_prm)
    print(scores1)



