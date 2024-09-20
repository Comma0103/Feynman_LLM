import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from call_gpt import Openai, API_INFOS

from mmlu_categories import categories, subcategories

from tqdm import tqdm

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    # s = ""
    # for entry in l:
    #     s += " " + entry
    s = " ".join(l)
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0] # question
    k = df.shape[1] - 2 # no. of choices
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1]) # choices
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1]) # answer
    return prompt

def gen_concept_request(df, subject, i):
    q_with_a = format_example(df, i)
    request = (
        "Assume you are a teacher or an LLM trainer, tasked with teaching about {0}.\n\n" + \
        "There is a multiple choice question (with answer) about {0}:\n" + \
        "{1}" + \
        "Based on your understanding of this question and {0}, " + \
        "could you briefly and clearly extract few central concepts covered in the question to teach someone who is not familiar with {0}, " + \
        "or a smaller language model, so that they can answer this question? " + \
        "Just simply list the extracted concepts (separated by comma and space, uppercase for the first letter) themselves only within 64 tokens, " + \
        "DO NOT include the answer:" # DO NOT number or explain them, and  or use any font format
    ).format(format_subject(subject), q_with_a)
    return request

@torch.no_grad()
def gen_concept(args, teacher_client, data_path, concept_path):
    # df.columns = ['questions', 'A', 'B', 'C', 'D', 'answers']
    # concepts = []
    # for i in tqdm(range(df.shape[0])):
    #     # get prompt and make sure it fits
    #     concept = teacher_client.call(gen_concept_request(df, subject, i))
    #     concepts.append(concept)
    # df['concepts'] = concepts

    # os.makedirs(os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name), df_names[df_idx]), exist_ok=True)
    # df.to_csv(
    #     os.path.join(
    #         args.concept_dir, "concepts_{}".format(args.concept_model_name), df_names[df_idx], subject + "_concepts.csv"
    #     ),
    #     index=False,
    # )
    
    with open(data_path, 'r', encoding='utf-8') as data_f:
        question_data_list = [json.loads(line) for line in data_f]
    with open(concept_path, 'w', encoding='utf-8') as concept_f:
        for question_data in tqdm(question_data_list):
            question = question_data['question']
            choices = question_data['choices']
            answer = question_data['answer']
            prompt = question
            for choice in choices:
                prompt += "\n{}".format(choice)
            prompt += "\nAnswer: {}".format(answer)
            concept = teacher_client.call(
                gen_concept_request(pd.DataFrame([[question] + choices + [answer]]), os.path.basename(data_path).split('-')[0], 0)
            )
            concept_f.write(json.dumps({
                "question": question,
                "choices": choices,
                "answer": answer,
                "concepts": concept
            }) + "\n")


def main(args):
    difficulty_levels = sorted(
        [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    ) # ['ARC-Challenge', 'ARC-Easy']
    
    oai_client = Openai(
        apis=API_INFOS[args.concept_model_name]
    )
    # res = oai_client.call("hello")

    if not os.path.exists(args.concept_dir):
        os.makedirs(args.concept_dir)
    if not os.path.exists(os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name))):
        os.makedirs(os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name)))

    for level in difficulty_levels:
        data_splits = ['Dev', 'Train', 'Test']
        for split in data_splits:
            print('Generating concepts for ', f'{level}-{split}', '...', sep='', end='\t')
            tic = time.time()
            
            data_path = os.path.join(args.data_dir, level, f'{level}-{split}.jsonl')
            concept_path = os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name), f'{level}-{split}.jsonl')
            gen_concept(args, oai_client, data_path, concept_path)
            
            toc = time.time()
            print("\nTime: {:.3f} s, {}, {}\n".format(toc-tic, level, split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/ARC-V1-Feb2018")
    parser.add_argument("--concept_dir", "-c", type=str, default="concepts/arc")
    # parser.add_argument("--concept_model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/xxx")
    parser.add_argument("--concept_model_name", "-cn", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    main(args)
