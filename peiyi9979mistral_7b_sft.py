import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


# 读取 JSON 文件
with open('questions.json', 'r') as file:
    data = json.load(file)

question_count = len(data)
print(f"JSON 中问题的条数为：{question_count}")


# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
model = AutoModelForCausalLM.from_pretrained("peiyi9979/mistral-7b-sft").eval()


# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = ("Chandra has four bowls.  Each one is a different color (red, blue, yellow, green).  She also has exactly"
          " one glass the same color as each bowl.  If she chooses a bowl and a glass from the cupboard, how many pa"
          "irings are possible?  One such pairing is a blue bowl and a yellow glass.")

# 调用模型回答问题
inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
outputs = model.generate(inputs, max_length=1024, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"模型生成的答案：{generated_text}")
