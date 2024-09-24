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


def format_example(question, choices, answer, include_answer=True):
    choices = sorted(choices, key=lambda x: x['label'])
    prompt = question # question
    k = len(choices) # no. of choices
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j]['label'], choices[j]['text']) # choices
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(answer) # answer
    return prompt

def gen_concept_request(question, choices, answer, level, grade):
    q_with_a = format_example(question, choices, answer)
    request = (
        "Assume you are a teacher or an LLM trainer, tasked with teaching about science exam questions that span several grades. " + \
        "Each question has a multiple choice structure (typically 4 answer options, some could have 3 or 5). " + \
        "The questions are sorted into a Challenge Set of 'hard' questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of questions.\n\n" + \
        "There is a multiple choice question (with answer) at grade {1} in the {0} set:\n\n" + \
        "{2}" + \
        "Based on your understanding of this question and the information above, " + \
        "could you briefly and clearly extract few central concepts covered in the question to teach someone who is not familiar with the relative knowledges, " + \
        "or a smaller language model, so that they can answer this question? " + \
        "Just simply list the extracted concepts (separated by comma and space, uppercase for the first letter) themselves only within 64 tokens, " + \
        "DO NOT include the answer:" # DO NOT number or explain them, and  or use any font format
    ).format(level, grade, q_with_a)
    return request

@torch.no_grad()
def gen_concept(args, teacher_client, data_path, concept_path):
    level = 'challenge' if 'Challenge' in data_path else 'easy' if 'Easy' in data_path else 'unknown'
    data_csv_path = data_path.replace('.jsonl', '.csv')
    
    df = pd.read_csv(data_csv_path)
    with open(data_path, 'r', encoding='utf-8') as data_f:
        question_data_list = [json.loads(line) for line in data_f]
    if len(df) != len(question_data_list):
        raise ValueError(f"The number of questions in the CSV file ({len(df)}) does not match the number of questions in the JSONL file ({len(question_data_list)}).")
    
    with open(concept_path, 'w', encoding='utf-8') as concept_f:
        for idx, question_data in enumerate(tqdm(question_data_list, ncols=75)):
            if df.iloc[idx, 0] != question_data['id']:
                raise ValueError(f"The question IDs of question #{idx+1} in the CSV file ({df.iloc[idx, 0]}) and the JSONL file ({question_data['id']}) do not match.")
            question = question_data['question']['stem']
            choices = question_data['question']['choices']
            answer = question_data['answerKey']
            grade = df.iloc[idx, 7]
            question_data['grade'] = str(grade)
            
            concept = teacher_client.call(gen_concept_request(question, choices, answer, level, grade))
            question_data['concepts'] = concept
            
            concept_f.write(json.dumps(question_data) + "\n")


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
            print('Generating concepts for ', f'{level}-{split}', '...', sep='', end='\n')
            tic = time.time()
            
            data_path = os.path.join(args.data_dir, level, f'{level}-{split}.jsonl')
            concept_path = os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name), f'{level}-{split}_concepts.jsonl')
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
