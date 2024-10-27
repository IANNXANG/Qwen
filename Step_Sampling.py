import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pprint




# 读取 JSONL 文件
with open('train.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]


question_count = len(data)
print(f"JSON 中问题的条数为：{question_count}")

model_path = "/pubshare/LLM/Qwen2.5-Math-1.5B-Instruct"
#cache_dir = "/pubshare/LLM"
# cache_dir = "/home/jovyan/.cache/huggingface/hub"
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).eval()
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct", cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct", cache_dir=cache_dir)

# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prm_model_path = "/pubshare/LLM/math-shepherd-mistral-7b-prm"
good_token = '+'
bad_token = '-'
step_tag = 'ки'
prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_path)
candidate_tokens = prm_tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
step_tag_id = prm_tokenizer.encode(f"{step_tag}")[-1]  # 12902
print("---------------------------------------------")
print("判断符:",candidate_tokens)
print("分隔符:",step_tag_id)
# [648, 387]
# 12902
print("---------------------------------------------")
prm_model = AutoModelForCausalLM.from_pretrained(model_path).eval()
prm_model.to(device)





# 打印读取到的 JSON 数据
for item in data:
    print("------------------------------------------------------------------------------------")
    print(f"问题：{item['problem']}\n答案：{item['answer']}")
    inputs = tokenizer(item['problem'] + "\n\n", return_tensors="pt").to(device)

    # 生成回答
    outputs = model.generate(**inputs, max_length=1000)
    # print(outputs[0])
    # print(outputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(answer)

    parts = answer.split("\n\n")
    result_dict = {}
    for index, part in enumerate(parts):
        key = f"step{index}" if index > 0 else "question"
        result_dict[key] = part
    print("------------------------------------------------------------------------------------")
    pprint.pprint(result_dict)
