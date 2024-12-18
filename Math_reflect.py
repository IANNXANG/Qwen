import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pprint




# 读取 JSONL 文件
with open('math/train.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]


question_count = len(data)
print(f"JSON 中问题的条数为：{question_count}")

model_path = "/pubshare/zy/cache/LLMfactory/Math_reflect"
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).eval()

# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 打印读取到的 JSON 数据
for item in data:
    print("------------------------------------------------------------------------------------")
    print(f"问题：{item['problem']}\n答案：{item['answer']}")
    inputs = tokenizer(item['problem'] + "\n\n", return_tensors="pt").to(device)

    with torch.no_grad():
    # 生成回答
        outputs = model.generate(**inputs, max_length=2048)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("="*30,"答案", "="*30)
    print(answer)
    print("="*30,"答案", "="*30)

    result_dict = {}
    result_dict["problem"] = item["problem"]
    result_dict["answer"] = answer
    with open('/home/jovyan/notebook/result.jsonl', 'a') as file:
        json.dump(result_dict, file)
        file.write('\n')
    print("------------------------------------------------------------------------------------")

