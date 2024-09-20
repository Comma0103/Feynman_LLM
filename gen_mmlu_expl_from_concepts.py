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


def gen_expl_request(df, i):
    concepts = df.loc[i, "concepts"]
    if pd.isna(concepts):
        return None
    request = (
        "Assume you are a teacher or an LLM trainer, tasked with teaching concepts of various disciplines.\n\n" + \
        "There are some related concepts of a question:\n" + \
        "{0}\n" + \
        "Based on your understanding of these concepts, " + \
        "could you briefly and clearly explain them to teach someone who is not familiar with them, " + \
        "or a smaller language model, so that they can answer the question? " + \
        "Just simply list and explain the concepts you extract within 256 tokens, DO NOT use any font format or extra words:"
    ).format(concepts)
    return request

def gen_expl(args, subject, teacher_client, dev_concepts_df, val_concepts_df, test_concepts_df):
    df_names = ['dev', 'val', 'test']
    for df_idx, df in enumerate([dev_concepts_df, val_concepts_df, test_concepts_df]):
        save_path = os.path.join(args.expl_dir, "expls_{}".format(args.expl_model_name), df_names[df_idx], subject + "_expls.csv")
        if os.path.exists(save_path):
            print(df_names[df_idx], ' already exists, skipping...', sep='', end=' ')
            continue
        
        print(df_names[df_idx], '..', sep='', end='\n')
        tic = time.time()
        
        expls = []
        for i in tqdm(range(df.shape[0])):
            # get prompt and make sure it fits
            request = gen_expl_request(df, i)
            expl = teacher_client.call(request) if request is not None else None
            expls.append(expl)
        df['explanations'] = expls
        
        os.makedirs(os.path.join(args.expl_dir, "expls_{}".format(args.expl_model_name), df_names[df_idx]), exist_ok=True)
        df.to_csv(save_path, index=False,)
        
        toc = time.time()
        print("time: {:.3f}s".format(toc-tic), end='\n')

def main(args):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    
    oai_client = Openai(
        apis=API_INFOS[args.expl_model_name]
    )
    # res = oai_client.call("hello")

    if not os.path.exists(args.expl_dir):
        os.makedirs(args.expl_dir)
    if not os.path.exists(os.path.join(args.expl_dir, "expls_{}".format(args.expl_model_name))):
        os.makedirs(os.path.join(args.expl_dir, "expls_{}".format(args.expl_model_name)))

    for subject in subjects:
        print('Generating explanations for ', subject, '...', sep='', end='\t')
        tic = time.time()
        
        dev_concepts_df = pd.read_csv(
            os.path.join(
                args.concept_dir,
                "concepts_{}".format(args.concept_model_name),
                "dev",
                subject + "_concepts.csv",
            )
        )
        val_concepts_df = pd.read_csv(
            os.path.join(
                args.concept_dir,
                "concepts_{}".format(args.concept_model_name),
                "val",
                subject + "_concepts.csv",
            )
        )
        test_concepts_df = pd.read_csv(
            os.path.join(
                args.concept_dir,
                "concepts_{}".format(args.concept_model_name),
                "test",
                subject + "_concepts.csv",
            )
        )

        gen_expl(args, subject, oai_client, dev_concepts_df, val_concepts_df, test_concepts_df)
        
        toc = time.time()
        print("\nTime: {:.3f} s, {}, {} of {}\n".format(toc-tic, subject, subjects.index(subject)+1, len(subjects)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/mmlu_data")
    parser.add_argument("--concept_dir", "-c", type=str, default="concepts/mmlu")
    parser.add_argument("--expl_dir", "-e", type=str, default="explanations/mmlu")
    # parser.add_argument("--concept_model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/xxx")
    parser.add_argument("--concept_model_name", "-cn", type=str, default="OpenAI-GPT-4o-mini")
    # parser.add_argument("--expl_model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/xxx")
    parser.add_argument("--expl_model_name", "-en", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    main(args)
