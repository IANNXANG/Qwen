from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

good_token = '+'
bad_token = '-'
step_tag = 'ки'

model_path = "/pubshare/LLM/math-shepherd-mistral-7b-prm"

tokenizer = AutoTokenizer.from_pretrained(model_path)
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902
model = AutoModelForCausalLM.from_pretrained(model_path).eval()

question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки"""  # 18 is right
output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""  # 17 is wrong
output3 = """Step 1: Janet's ducks lay 12 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""
output4 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 3: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""
output5 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nки"""
output6 = "Step 1: Janet's ducks lay 16 eggs per day. ки\nки"
output7 = "Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки"  # 18 is right

# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])  最后一个错了
# tensor([0.3700, 0.6413, 0.8014, 0.0163])  把第一个改错
# tensor([0.9955, 0.2233, 0.0090])   删除第二个
# tensor([0.9955])
# tensor([0.9955])
# tensor([0.9955, 0.9958, 0.9983, 0.9957])

oplist = []
for i in range(1, 8):
    output_name = f'output{i}'
    # 假设这里有获取对应 output 变量值的方法，这里只是模拟用字符串来代替真实的值
    value = eval(output_name)
    oplist.append(value)



for output in oplist:
    input_for_prm = f"{question} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])

    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens]
        scores = logits.softmax(dim=-1)[:, :, 0]
        step_scores = scores[input_id == step_tag_id]
        print(step_scores)

# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])
