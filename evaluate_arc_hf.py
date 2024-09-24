import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mmlu_categories import categories, subcategories

from tqdm import tqdm

# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True


def format_example(question, choices, answer, include_answer=True):
    choices = sorted(choices, key=lambda x: x['label'])
    prompt = question # question
    k = len(choices) # no. of choices
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j]['label'], choices[j]['text']) # choices
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(answer) # answer
    # if include_answer:
    #     prompt += "\nAnswer:"
    #     prompt += " {}\n\n".format(answer) # answer
    # else:
    #     prompt += "\nAnswer (choosing from A, B, C, and D):"
    return prompt

def gen_few_shot_prompt(dev_data_list, level, k=-1):
    prompt = f"The following are multiple choice questions (with answers) in {level} dataset.\n\n"
    if k == -1:
        k = 5 # no. of questions
    for i in range(k):
        random_idx = np.random.randint(len(dev_data_list))
        question = dev_data_list[random_idx]['question']['stem']
        choices = dev_data_list[random_idx]['question']['choices']
        answer = dev_data_list[random_idx]['answerKey']
        prompt += format_example(question, choices, answer)
    return prompt

        
@torch.no_grad()
def eval(args, level, model, tokenizer, dev_data_list, test_data_list):
    cors = []
    all_probs = []
    
    for i in tqdm(range(len(test_data_list)), ncols=75):
        question = test_data_list[i]['question']['stem']
        choices = test_data_list[i]['question']['choices']
        answer = test_data_list[i]['answerKey']
        # get prompt and make sure it fits
        k = args.ndev
        prompt_end = "Now, you will answer the following question:\n"
        prompt_end += format_example(question, choices, None, include_answer=False)
        few_shot_prompt = gen_few_shot_prompt(dev_data_list, level, k)
        prompt = few_shot_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        while input_ids.shape[-1] > 2048 and k > 0:
            k -= 1
            few_shot_prompt = gen_few_shot_prompt(dev_data_list, level, k)
            prompt = few_shot_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        label = answer

        logits = model(input_ids=input_ids).logits[0, -1]

        choice_labels = sorted([choice['label'] for choice in choices])
        probs = (
            torch.nn.functional.softmax(
                torch.tensor([logits[tokenizer(choice_label).input_ids[-1]] for choice_label in choice_labels]).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = choice_labels[np.argmax(probs)]
        probs_dict = {choice_labels[i]: probs[i] for i in range(len(choice_labels))}

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs_dict)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.6f} - {}".format(acc, level))

    return cors, acc, all_probs


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    
    difficulty_levels = sorted(
        [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    ) # ['ARC-Challenge', 'ARC-Easy']
    data_splits = ['Dev', 'Train', 'Test']

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.exp_name))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.exp_name)))

    all_cors = []
    level_split_cors = {f'{level}-{split}': [] for level in difficulty_levels for split in data_splits}
    level_cors = {f'{level}': [] for level in difficulty_levels}
    
    for level in difficulty_levels:
        tic = time.time()
        
        with open(os.path.join(args.data_dir, level, f'{level}-Dev.jsonl'), 'r', encoding='utf-8') as dev_f:
            dev_data_list = [json.loads(line) for line in dev_f]
        with open(os.path.join(args.data_dir, level, f'{level}-Test.jsonl'), 'r', encoding='utf-8') as test_f:
            test_data_list = [json.loads(line) for line in test_f]

        cors, acc, probs = eval(args, level, model, tokenizer, dev_data_list, test_data_list)
        level_cors[level].append(cors)
        level_split_cors[f'{level}-Test'] = cors
        all_cors.append(cors)

        for i in range(len(cors)):
            test_data_list[i]["{}_correct".format(args.model_name)] = cors[i]
            test_data_list[i]["{}_choice_probs".format(args.model_name)] = probs[i]
        with open(os.path.join(args.save_dir, "results_{}".format(args.exp_name), f'{level}-Test_results.jsonl'), 'w', encoding='utf-8') as out_f:
            for test_data in test_data_list:
                out_f.write(json.dumps(test_data) + "\n")
        toc = time.time()
        print("\tTime: {:.3f} s, {} of {}\n".format(toc-tic, difficulty_levels.index(level)+1, len(difficulty_levels)))

    results = {"level": {}, "level_split": {}}
    print("Level Split accuracies:")
    for level_split in level_split_cors:
        level_split_acc = np.mean(level_split_cors[level_split])
        results["level_split"][level_split] = level_split_acc
        print("Average accuracy {:.3f} - {}".format(level_split_acc, level_split))
    print("\nLevel accuracies:")
    for level in level_cors:
        level_acc = np.mean(np.concatenate(level_cors[level]))
        results["level"][level] = level_acc
        print("Average accuracy {:.3f} - {}".format(level_acc, level))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("\nTotal Average accuracy: {:.3f}".format(weighted_acc))
    print(args.exp_name)

    results_file = os.path.join(
        args.save_dir, "accuracies_{}.json".format(args.exp_name)
    )
    with open(results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndev", "-k", type=int, default=5, help="Number of few-shot questions in the prompt")
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/ARC-V1-Feb2018")
    parser.add_argument("--save_dir", "-s", type=str, default="results/arc")
    parser.add_argument("--model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/Meta-Llama-3-8B")
    parser.add_argument("--model_name", "-n", type=str, default="Meta-Llama-3-8B")
    args = parser.parse_args()
    args.exp_name = f"{args.model_name}"
    main(args)
