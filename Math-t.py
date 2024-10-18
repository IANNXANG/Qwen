from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


good_token = '+'
bad_token = '-'
step_tag = 'ки'

model_path = "/pubshare/LLM/math-shepherd-mistral-7b-prm"

tokenizer = AutoTokenizer.from_pretrained(model_path)
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902
print("---------------------------------------------")
print(candidate_tokens)
print(step_tag_id)
# [648, 387]
# 12902
print("---------------------------------------------")

model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model.to(device)

question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки"""  # 18 is right
output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""  # 17 is wrong
output3 = """Step 1: Janet's ducks lay 12 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""
output4 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 3: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""
oplist = []
n = 4
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])  最后一个错了
# tensor([0.3700, 0.6413, 0.8014, 0.0163])  把第一个改错
# tensor([0.9955, 0.2233, 0.0090])   删除第二个




for i in range(1, n+1):
    output_name = f'output{i}'
    # 假设这里有获取对应 output 变量值的方法，这里只是模拟用字符串来代替真实的值
    value = eval(output_name)
    oplist.append(value)



for output in oplist:
    input_for_prm = f"{question} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])

    inputs = tokenizer(input_for_prm, return_tensors="pt").to(device)

    # 生成回答
    outputs = model.generate(**inputs, max_length=1000)
    # print(outputs[0])
    # print(outputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(answer)


# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])
