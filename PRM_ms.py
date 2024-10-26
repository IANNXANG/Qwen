from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

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

token_id = 1107
decoded_text = tokenizer.decode([token_id])
print("1107的token:")
print(decoded_text)

token2_id = 12902
decoded_text2 = tokenizer.decode([token2_id])
print("12902的token:")
print(decoded_text2)

print("---------------------------------------------")





model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model.to(device)

question = """
Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\n
Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\n
Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\n
Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. 
The answer is: 18 ки"""  # 18 is right
output2 = """
Step 1: Janet's ducks lay 16 eggs per day. ки\n
Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\n
Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\n
Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. 
The answer is: 17 ки"""  # 17 is wrong
output3 = """Step 1: Janet's ducks lay 12 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""
output4 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 3: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""
oplist = []
n = 2
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])  把最后一个step改错
# tensor([0.3700, 0.6413, 0.8014, 0.0163])  把第一个改错
# tensor([0.9955, 0.2233, 0.0090])          删除第二个step


#在q和a之间添加\n
#tensor([0.9929, 0.9956, 0.9959, 0.9934], device='cuda:0')
#tensor([0.9929, 0.9956, 0.9959, 0.0280], device='cuda:0')
#tensor([0.3201, 0.5929, 0.7085, 0.0187], device='cuda:0')
#tensor([0.9929, 0.2399, 0.0108], device='cuda:0')


for i in range(1, n+1):
    output_name = f'output{i}'
    # 假设这里有获取对应 output 变量值的方法，这里只是模拟用字符串来代替真实的值
    value = eval(output_name)
    oplist.append(value)

# Bug 修复：在使用 tokenized_result 之前，确保它已经被定义
for output in oplist:
    input_for_prm = f"{question}\n{output}"
    print("---------------------------------------------")
    print("input_for_prm:")
    print(input_for_prm)
    # 获取分词结果但不进行编码
    tokenized_result = tokenizer.tokenize(input_for_prm)
    print("---------------------------------------------")
    print("Tokenized result:")
    print(tokenized_result)

    # Bug 修复：将 tokenized_result 转换为 Tensor 类型
    tokenized_result = torch.tensor(tokenized_result).to(device)

    input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)
    print("---------------------------------------------")
    print("input_id:")
    #print(input_id)

    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens].to(device)
        print("---------------------------------------------")
        # print("logits:")
        # print(logits)   #输出为+或者-的logits
        scores = logits.softmax(dim=-1)[:, :, 0].to(device)
        print("---------------------------------------------")
        print("scores:")
        print(scores)   #输出为+的softmax
        step_scores = scores[input_id == step_tag_id].to(device)
        print("---------------------------------------------")
        print("step_scores:")
        print(step_scores)


