import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from call_gpt import Openai, API_INFOS

from tqdm import tqdm

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


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
    random_indices = np.random.choice(len(dev_data_list), k, replace=False)
    for random_idx in random_indices:
        question = dev_data_list[random_idx]['question']['stem']
        choices = dev_data_list[random_idx]['question']['choices']
        answer = dev_data_list[random_idx]['answerKey']
        prompt += format_example(question, choices, answer)
    return prompt

# def gen_2nd_expl_request(concept_with_taxo_path, old_concept_expl):
#     request = (
#         "Assume you are a teacher or an LLM trainer, tasked with teaching concepts of various disciplines about science exam questions that span several grades.\n\n" + \
#         "There is a concept(left part of ':') with its path in the human knowledge taxonomy system(right part of ':'):\n" + \
#         "{0}\n" + \
#         "where the 4 levels in the taxonomy path are 'discipline -> subject -> class session -> knowledge point' if the path is found, otherwise, understand it according to your knowledge.\n\n" + \
#         "You used to give the explanation of this concept as follow:\n" + \
#         "{1}\n\n" + \
#         "However, a student LM got a wrong answer after adding your explanation into its prompt when answering the question that this concept related to.\n\n" + \
#         "Based on your understanding of this concept, think about the reason why the student got the wrong answer, " + \
#         "could you modify the explanation of this concept to teach anyone who is not familiar with it (not only this student LM), " + \
#         "or a smaller language model, so that they can understand it easily then answer the question right?\n\n" + \
#         "Just simply explain the concept itself (NOT knowledge point) using the format 'concept: explanation' within 32 tokens, DO NOT use any font format or extra words:"
#     ).format(concept_with_taxo_path, old_concept_expl)
#     return request

def gen_expl_prompt(concepts=None, taxonomy_paths=None, new_explanations=None, explanations=None):
    if not pd.isna(concepts) and not pd.isna(taxonomy_paths) and (not pd.isna(new_explanations) or not pd.isna(explanations)):
        expl_prompt = "Here are some concepts with their path in the taxonomy system (if not found, understand according to your knowledge), " + \
                "and explanations of the concepts (if not available, understand according to your knowledge), which may be useful:\n"
        concepts_list = concepts.split(", ")
        taxonomy_paths_list = [e.split(": ")[1] if ": " in e else e for e in taxonomy_paths.split(";|; ")]
        explanations_used = new_explanations if not pd.isna(new_explanations) else explanations
        explanations_list = [e.split(": ")[1] if ": " in e else e for e in explanations_used.split(";|; ")]
        for concept, taxo_path, expl in zip(concepts_list, taxonomy_paths_list, explanations_list):
            expl_prompt += "Concept: {}, Taxonomy Path: {}, Explanation: {}\n".format(concept, taxo_path, expl)
        expl_prompt += "\n"
    elif not pd.isna(concepts) and not pd.isna(taxonomy_paths):
        expl_prompt = "Here are some concepts with their path in the taxonomy system (if not found, understand according to your knowledge), " + \
                "which may be useful:\n"
        concepts_list = concepts.split(", ")
        taxonomy_paths_list = [e.split(": ")[1] if ": " in e else e for e in taxonomy_paths.split(";|; ")]
        for concept, taxo_path in zip(concepts_list, taxonomy_paths_list):
            expl_prompt += "Concept: {}, Taxonomy Path: {}.\n".format(concept, taxo_path)
        expl_prompt += "\n"
    elif not pd.isna(concepts):
        expl_prompt = "Here are some concepts that may be useful:\n"
        expl_prompt += concepts
        expl_prompt += ".\n\n"
    else:
        expl_prompt = ""
    return expl_prompt
        
@torch.no_grad()
def forward(args, model, tokenizer, input_ids, choices):
    logits = model(input_ids=input_ids).logits[0, -1]

    choice_labels = sorted([str(choice['label']) for choice in choices])
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
    probs_dict = {choice_labels[i]: str(probs[i]) for i in range(len(choice_labels))}
    return probs_dict, pred

def eval(args, level, model, tokenizer, dev_data_list, test_data_list, dev_expl_list, test_expl_list):
    cors = []
    all_probs = []
    # all_new_expls = []
    
    if len(dev_data_list) != len(dev_expl_list):
        raise ValueError(f"The number of questions in the dev data file ({len(dev_data_list)}) does not match with the dev expl file ({len(dev_expl_list)}).")
    if len(test_data_list) != len(test_expl_list):
        raise ValueError(f"The number of questions in the test data file ({len(test_data_list)}) does not match with the test expl file ({len(test_expl_list)}).")

    for i in tqdm(range(len(test_data_list)), ncols=75):
        question = test_data_list[i]['question']['stem']
        choices = test_data_list[i]['question']['choices']
        answer = test_data_list[i]['answerKey']
        # get prompt and make sure it fits
        k = args.ndev
        prompt_end = "Now, you will answer the following question:\n"
        prompt_end += format_example(question, choices, None, include_answer=False)
        few_shot_prompt = gen_few_shot_prompt(dev_data_list, level, k)
        expl_prompt = gen_expl_prompt(
            test_expl_list[i]["concepts"],
            test_expl_list[i]["taxonomy_path"],
            test_expl_list[i]["new_explanations"],
            test_expl_list[i]["explanations"]
        )
        prompt = expl_prompt + few_shot_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        while input_ids.shape[-1] > 2048 and k > 0:
            k -= 1
            few_shot_prompt = gen_few_shot_prompt(dev_data_list, level, k)
            prompt = expl_prompt + few_shot_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        label = str(answer)

        probs, pred = forward(args, model, tokenizer, input_ids, choices)
        cor = pred == label
        
        # if not cor:
        #     if pd.isna(test_expl_list[i]["taxonomy_path"]) or test_expl_list[i]["taxonomy_path"] == "":
        #         new_expls = None
        #         all_new_expls.append(new_expls)
        #     else:
        #         concepts = test_expl_list[i]["concepts"].split(", ")
        #         concepts_with_taxo_path = test_expl_list[i]["taxonomy_path"].split(";|; ")
        #         if pd.isna(test_expl_list[i]["explanations"]):
        #             old_expls = [concept + ": Not available" for concept in concepts]
        #         else:
        #             old_expls = test_expl_list[i]["explanations"].split(";|; ")
        #         new_expls = []
        #         for concept, concept_with_taxo_path, old_expl in zip(concepts, concepts_with_taxo_path, old_expls):
        #             if "Not found" in concept_with_taxo_path:
        #                 concept_with_taxo_path = concept + ": Not found"
        #             # get prompt and make sure it fits
        #             new_expl = teacher_client.call(gen_2nd_expl_request(concept_with_taxo_path, old_expl)) 
        #             new_expls.append(new_expl if new_expl is not None else concept + ": Not available")
        #         all_new_expls.append(";|; ".join(new_expls))
            
        #     expl_prompt = gen_expl_prompt(
        #         test_expl_list[i]["concepts"],
        #         test_expl_list[i]["taxonomy_path"],
        #         ";|; ".join(new_expls) if new_expls is not None else None
        #     )
        #     prompt = expl_prompt + few_shot_prompt + prompt_end
        #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        #     probs, pred = forward(args, model, tokenizer, input_ids, choices)
        #     cor = pred == label
        # else:
        #     all_new_expls.append(None)
        
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.6f} - {}".format(acc, level))

    return cors, acc, all_probs#, all_new_expls


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
    
    # explanations have been generated in expls_OpenAI-GPT-4o-mini_sep_taxo_path_gen_Meta-Llama-3-8B_wo_choosing_2nd
    # oai_client = Openai(
    #     apis=API_INFOS[args.expl_model_name]
    # )
    # res = oai_client.call("hello")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.exp_name))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.exp_name)))
    # if not os.path.exists(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}_{}_wo_choosing_2nd_use_llama3-8b_expl".format(args.expl_model_name, args.taxo_path_src, args.model_name))):
    #     os.makedirs(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}_{}_wo_choosing_2nd_use_llama3-8b_expl".format(args.expl_model_name, args.taxo_path_src, args.model_name)))

    all_cors = []
    level_split_cors = {f'{level}-{split}': [] for level in difficulty_levels for split in data_splits}
    level_cors = {f'{level}': [] for level in difficulty_levels}

    for level in difficulty_levels:
        tic = time.time()
        
        with open(os.path.join(args.data_dir, level, f'{level}-Dev.jsonl'), 'r', encoding='utf-8') as dev_f:
            dev_data_list = [json.loads(line) for line in dev_f]
        with open(os.path.join(args.data_dir, level, f'{level}-Test.jsonl'), 'r', encoding='utf-8') as test_f:
            test_data_list = [json.loads(line) for line in test_f]
        with open(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}".format(args.expl_model_name, args.taxo_path_src), f'{level}-Dev_expls.jsonl'), 'r', encoding='utf-8') as dev_expl_f:
            dev_expl_list = [json.loads(line) for line in dev_expl_f]
        with open(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}_Meta-Llama-3-8B_wo_choosing_2nd".format(args.expl_model_name, args.taxo_path_src), f'{level}-Test_2nd_expls.jsonl'), 'r', encoding='utf-8') as test_expl_f:
            test_expl_list = [json.loads(line) for line in test_expl_f]

        # cors, acc, probs, all_new_expls = eval(args, level, model, tokenizer, dev_data_list, test_data_list, dev_expl_list, test_expl_list, oai_client)
        cors, acc, probs = eval(args, level, model, tokenizer, dev_data_list, test_data_list, dev_expl_list, test_expl_list)        
        level_cors[level].append(cors)
        level_split_cors[f'{level}-Test'] = cors
        all_cors.append(cors)

        for i in range(len(cors)):
            test_data_list[i]["{}_correct".format(args.model_name)] = bool(cors[i])
            test_data_list[i]["{}_choice_probs".format(args.model_name)] = probs[i]
        with open(os.path.join(args.save_dir, "results_{}".format(args.exp_name), f'{level}-Test_results.jsonl'), 'w', encoding='utf-8') as out_f:
            for test_data in test_data_list:
                out_f.write(json.dumps(test_data) + "\n")
        # for i in range(len(all_new_expls)):
        #     test_expl_list[i]["new_explanations"] = all_new_expls[i]
        # with open(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}_{}_wo_choosing_2nd".format(args.expl_model_name, args.taxo_path_src, args.model_name), f'{level}-Test_2nd_expls.jsonl'), 'w', encoding='utf-8') as out_f:
        #     for test_expl in test_expl_list:
        #         out_f.write(json.dumps(test_expl) + "\n")
        
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
    parser.add_argument("--ndev", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/ARC-V1-Feb2018")
    parser.add_argument("--expl_dir", "-e", type=str, default="explanations/arc")
    parser.add_argument("--save_dir", "-s", type=str, default="results/arc")
    # parser.add_argument("--model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/Meta-Llama-2-7B-hf")
    parser.add_argument("--model_path", "-m", type=str, default="/data/qilongma/public_models/Meta-Llama-2-7B-hf")
    parser.add_argument("--model_name", "-n", type=str, default="Meta-Llama-2-7B")
    parser.add_argument("--taxo_path_src", "-tp", type=str, default="gen", choices=["gen", "search"])
    parser.add_argument("--expl_model_name", "-en", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    args.exp_name = f"{args.model_name}_feynman_{args.expl_model_name}_sep_question_concept_taxo_path_{args.taxo_path_src}_wo_choosing_2nd_use_llama3-8b_expl"
    main(args)
