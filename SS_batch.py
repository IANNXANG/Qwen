import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pprint

# 读取 JSONL 文件
with open('train.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

# 设置批量大小
batch_size = 16
model_path = "/pubshare/zy/cache/Qwen2.5-Math-1.5B-Instruct"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 打印读取到的 JSON 数据
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    print("------------------------------------------------------------------------------------")
    print(f"Processing batch {i // batch_size + 1}")

    inputs = tokenizer([item['problem'] + "\n\n" for item in batch], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        # 生成回答
        # bug修复：移除batch_size参数
        outputs = model.generate(**inputs, max_length=1000)
        answers = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    for j, answer in enumerate(answers):
        print(f"问题：{batch[j]['problem']}\n答案：{answer}")

        parts = answer.split("\n\n")
        result_dict = {}
        for index, part in enumerate(parts):
            key = f"step{index}" if index > 0 else "question"
            result_dict[key] = part
        with open('/home/jovyan/notebook/zhouyang/result16.jsonl', 'a') as file:
            json.dump(result_dict, file)
            file.write('\n')

    print("------------------------------------------------------------------------------------")
    pprint.pprint(result_dict)
