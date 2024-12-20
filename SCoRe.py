import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch.optim as optim
import argparse
from PRM_ms import calculate_step_scores
import numpy as np


#prm返回分数 保存为一个list
def get_scores(input_for_prm:str)->list:
    return calculate_step_scores(input_for_prm, prm_model, prm_tokenizer, device1, candidate_tokens, step_tag_id)

def GenAndScore(prompt:str):
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

def get_input_for_prm(result_dict:dict)->str:
    input_for_prm = ""
    for i, key in enumerate(result_dict):
        if i == 0:
            input_for_prm = input_for_prm + result_dict[key]
        else:
            input_for_prm = input_for_prm + result_dict[key] + 'ки'
    #skip_special_tokens=False时使用这一行代码
    #input_for_prm = input_for_prm.replace("<|im_end|>", "")
    return input_for_prm

def get_result_dict(answer:str)->dict:
    parts = answer.split("\n\n")
    result_dict = {}
    for index, part in enumerate(parts):
        key = f"step{index}" if index > 0 else "question"
        result_dict[key] = part
    print(json.dumps(result_dict, indent=4))
    #pprint.pprint(result_dict)  #打印功能
    return result_dict

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

prm_model = AutoModelForCausalLM.from_pretrained(model_path)
prm_model.to(device1)


#Actor模型放置在GPU1上
# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")

# 设置模型运行环境
#创建ref模型
model_ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
tokenizer_ref = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")


device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device2)
model_ref.to(device2)
model.train() # 启用训练模式

max_length = 1024



# 打印读取到的 JSON 数据
for item in data[:10]:
    print("------------------------------------------------------------------------------------")
    print(f"问题：{item['problem']}\n答案：{item['answer']}")
    inputs = tokenizer(item['problem'] + "\n\n", return_tensors="pt").to(device2)
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    # print("outputs[0]：\n",outputs[0])
    # print("outputs：\n",outputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


    #答案保存为字典
    result_dict = get_result_dict(answer)

    #创建input_for_prm第一次
    input_for_prm = get_input_for_prm(result_dict)


    #使用input_for_prm生成分数
    scores1 = get_scores(input_for_prm).tolist()
    print("=" * 30 + "分数1" + "=" * 30)
    print("scores1：", scores1)

    #创建反思的input
    print("="*30 + "添加反思" + "="*30)
    reflect_text = "Wait, did I make a mistake somewhere? Let me check again?"
    print("reflect_text：", reflect_text)
    input_for_sc_text = answer + "\n\n" + reflect_text + "\n\n"
    input_for_sc = tokenizer(input_for_sc_text, return_tensors="pt").to(device2)
    q = input_for_sc
    outputs_for_sc = model.generate(**input_for_sc, max_new_tokens=max_length)
    a = outputs_for_sc[:q.shape[1]:]  #裁剪
    answer_for_sc = tokenizer.decode(outputs_for_sc[0], skip_special_tokens=True)
    print("=" * 30 + "生成反思结果" + "=" * 30)
    result_dict2 = get_result_dict(answer_for_sc)
    input_for_prm2 = get_input_for_prm(result_dict2)
    scores2 = get_scores(input_for_prm2).tolist()
    scores2_only = scores2[len(scores1)+1:]
    print("=" * 30 + "分数1、分数2和分数2_only" + "=" * 30)
    print(f"反思提示是第 {len(scores1)+1} 步")
    print("scores1：", scores1)
    print("scores2：", scores2)
    print("scores2_only：", scores2_only)
    print("=" * 30 + "分数1和分数2_only的平均值" + "=" * 30)
    value1 = sum(scores1)/len(scores1)
    value2_only = sum(scores2_only)/len(scores2_only)
    print("value1：", value1)
    print("value2：", value2_only)
    print("=" * 30 + "计算损失" + "=" * 30)
    reward = value1 + value2_only
    r = [reward]



#保存模型和分词器
model.save_pretrained("/pubshare/zy/model/trained_self_correcting_model")
tokenizer.save_pretrained("/pubshare/zy/model/trained_self_correcting_model")