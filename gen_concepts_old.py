import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from call_gpt import Openai, API_INFOS

from categories import categories, subcategories

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
        "could you briefly and clearly extract the concepts covered in the question to teach someone who is not familiar with {0}, " + \
        "or a smaller language model, so that they can answer this question? " + \
        "Just simply list and explain the concepts you extract within 256 tokens, DO NOT include the answer or use any font format:"
    ).format(format_subject(subject), q_with_a)
    return request

@torch.no_grad()
def gen_concept(args, subject, teacher_client, dev_df, val_df, test_df):
    df_names = ['dev', 'val', 'test']
    for df_idx, df in enumerate([dev_df, val_df, test_df]):
        print(df_names[df_idx], '..', sep='', end=' ')
        tic = time.time()
        
        df.columns = ['questions', 'A', 'B', 'C', 'D', 'answers']
        concepts = []
        for i in tqdm(range(df.shape[0])):
            # get prompt and make sure it fits
            concept = teacher_client.call(gen_concept_request(df, subject, i))
            concepts.append(concept)
        df['concepts'] = concepts
        
        os.makedirs(os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name), df_names[df_idx]), exist_ok=True)
        df.to_csv(
            os.path.join(
                args.concept_dir, "concepts_{}".format(args.concept_model_name), df_names[df_idx], subject + "_concepts.csv"
            ),
            index=False,
        )
        
        toc = time.time()
        print("time: {:.3f}s".format(toc-tic), end='  ')

def main(args):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    
    oai_client = Openai(
        apis=API_INFOS[args.concept_model_name]
    )
    # res = oai_client.call("hello")

    if not os.path.exists(args.concept_dir):
        os.makedirs(args.concept_dir)
    if not os.path.exists(os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name))):
        os.makedirs(os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name)))

    for subject in subjects:
        print('Generating concepts for ', subject, '...', sep='', end='\t')
        tic = time.time()
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )
        val_df = pd.read_csv(
            os.path.join(args.data_dir, "val", subject + "_val.csv"), header=None
        )
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        gen_concept(args, subject, oai_client, dev_df, val_df, test_df)
        
        toc = time.time()
        print("\nTime: {:.3f} s, {}, {} of {}\n".format(toc-tic, subject, subjects.index(subject)+1, len(subjects)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/mmlu_data")
    parser.add_argument("--concept_dir", "-s", type=str, default="concepts")
    # parser.add_argument("--concept_model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/xxx")
    parser.add_argument("--concept_model_name", "-cn", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    main(args)
