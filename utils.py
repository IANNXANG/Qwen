from transformers import AutoTokenizer, AutoModelForCausalLM
import pprint

def PrintQandA(prompt,tokenizer,model):
    # 可以添加一些示例输入进行测试

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 生成回答
    outputs = model.generate(**inputs, max_length=1000)
    # print(outputs[0])
    # print(outputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=False)


    parts = answer.split("\n\n")
    result_dict = {}
    for index, part in enumerate(parts):
        key = f"step{index}" if index > 0 else "question"
        result_dict[key] = part
    print("------------------------------------------------------------------------------------")
    input("Press Enter to continue:")
    pprint.pprint(result_dict)