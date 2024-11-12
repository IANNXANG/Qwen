from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch



def calculate_step_scores(input_for_prm, model, tokenizer, device, candidate_tokens,step_tag_id):
    input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)

    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens].to(device)
        scores = logits.softmax(dim=-1)[:, :, 0].to(device)
        step_scores = scores[input_id == step_tag_id].to(device)

    return step_scores


if __name__ == '__main__':

    good_token = '+'
    bad_token = '-'
    step_tag = 'ки'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "/pubshare/LLM/math-shepherd-mistral-7b-prm"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1]

    model = AutoModelForCausalLM.from_pretrained(model_path).eval()
    model.to(device)



    # 使用示例
    question = """
    Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    output = """Step 1: Janet's ducks lay 16 eggs per day. ки\n
    Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\n
    Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\n
    Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. 
    The answer is: 18 ки"""

    input_for_prm = f"{question}\n{output}"
    step_scores = calculate_step_scores(input_for_prm, model, tokenizer, device, candidate_tokens,step_tag_id)
    print("Step Scores:", step_scores)
