import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json







# 初始化一个空列表来存储数据
data_list = []

# 读取jsonl文件
with open('result.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的JSON数据
        data = json.loads(line)
        data_list.append(data)

question_count = len(data_list)
print(f"JSON 中问题的条数为：{question_count}")

good_token = '+'
bad_token = '-'
step_tag = 'ки'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/pubshare/LLM/math-shepherd-mistral-7b-prm"

tokenizer = AutoTokenizer.from_pretrained(model_path)
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902
print("---------------------------------------------")
print("判断符:",candidate_tokens)
print("分隔符:",step_tag_id)
# [648, 387]
# 12902
print("---------------------------------------------")
model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model.to(device)


for index, data in enumerate(data_list):
    print(f"Data {index + 1}:")
    input_for_prm = ""
    for i, key in enumerate(data):
        print(f"{key}: {data[key]}")
        if i == 0:
            input_for_prm = input_for_prm + data[key]
        else:
            input_for_prm = input_for_prm + data[key] + 'ки'
    input_for_prm = input_for_prm.replace("<|im_end|>", "")
    print(input_for_prm)

    input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)
    print("---------------------------------------------")


    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens].to(device)
        scores = logits.softmax(dim=-1)[:, :, 0].to(device)
        step_scores = scores[input_id == step_tag_id].to(device)
        print("---------------------------------------------")
        print("step_scores:")
        print(step_scores)








    #     if index > 0:
    #         Input4PRM = Input4PRM + part + "ки"
    #     else:
    #         Input4PRM = Input4PRM + part
    # input_for_prm = Input4PRM
    # print("input_for_prm:",input_for_prm)
    # input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)
    # with torch.no_grad():
    #     logits = prm_model(input_id).logits[:, :, candidate_tokens].to(device)
    #     scores = logits.softmax(dim=-1)[:, :, 0].to(device)
    #     step_scores = scores[input_id == step_tag_id].to(device)
    #     print("---------------------------------------------")
    #     print("step_scores:")
    #     print(step_scores)

