import argparse
import json
import os
import time

import numpy as np
import pandas as pd

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
    k = len(choices) # no. of choices
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1]) # choices
    if include_answer:
        prompt += "\nAnswer:"
        prompt += " {}\n\n".format(df.iloc[idx, k + 1]) # answer
    else:
        prompt += "\nAnswer (choosing from A, B, C, and D):"
    return prompt

def gen_concept_request(df, subject, i):
    q_with_a = format_example(df, i)
    request = (
        "Assume that you are tasked with teaching about {0}.\n\n" + \
        "There is a multiple choice question (with answer) about {0}:\n" + \
        "{1}" + \
        "Based on your understanding of this question and {0}, " + \
        "could you briefly and clearly extract the concepts covered in the question to teach someone who is not familiar with {0}, " + \
        "or a smaller language model, so that they can answer this question? " + \
        "Just simply list and explain the concepts you extract within 256 tokens, DO NOT include the answer or use any font format:"
    ).format(format_subject(subject), q_with_a)
    return request

def gen_ftdata(args, subject, dev_concepts_df, val_concepts_df, test_concepts_df, all_data_f=None):
    df_names = ['dev', 'val', 'test']
    for df_idx, df in enumerate([dev_concepts_df, val_concepts_df, test_concepts_df]):
        if df is None:
            continue
        print(df_names[df_idx], '..', sep='', end=' ')
        tic = time.time()
        
        dialogs = []
        for i in (range(df.shape[0])):
            if pd.isna(df.loc[i, "new_concepts"]):
                continue
            request = gen_concept_request(df, subject, i)
            concept = df.loc[i, "new_concepts"]
            dialog = {
                "messages": [
                    {"role": "system", "content": "You are a teacher or an LLM trainer."},
                    {"role": "user", "content": request},
                    {"role": "assistant", "content": concept}
                ]
            }
            dialogs.append(dialog)
        
        os.makedirs(os.path.join(args.save_dir, "ftdata_{}_old_2nd".format(args.concept_model_name), df_names[df_idx]), exist_ok=True)
        data_file = os.path.join(
            args.save_dir, "ftdata_{}_old_2nd".format(args.concept_model_name), df_names[df_idx], subject + "_ftdata.jsonl"
        )
        with open(data_file, 'w', encoding="utf-8") as f:
            for dialog in dialogs:
                json_record = json.dumps(dialog)
                f.write(json_record + "\n")
                if all_data_f:
                    all_data_f.write(json_record + "\n")
        
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
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "ftdata_{}_old_2nd".format(args.concept_model_name))):
        os.makedirs(os.path.join(args.save_dir, "ftdata_{}_old_2nd".format(args.concept_model_name)))
    all_data_f = open(os.path.join(
        args.save_dir, "ftdata_{}_old_2nd".format(args.concept_model_name), "ftdata_{}_2nd.jsonl".format(args.concept_model_name)), "w", encoding="utf-8")

    for subject in subjects:
        print('Generating finetune data for ', subject, '...', sep='', end='\t')
        tic = time.time()

        dev_concepts_df = None
        val_concepts_df = None
        test_concepts_df = pd.read_csv(
            os.path.join(
                args.concept_dir,
                "concepts_{}_2nd".format(args.concept_model_name),
                "test",
                subject + "_2nd_concepts.csv",
            )
        )

        gen_ftdata(args, subject, dev_concepts_df, val_concepts_df, test_concepts_df, all_data_f)
        
        toc = time.time()
        print("\nTime: {:.3f} s, {} of {}\n".format(toc-tic, subjects.index(subject)+1, len(subjects)))
    
    all_data_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/mmlu_data")
    parser.add_argument("--concept_dir", "-c", type=str, default="concepts/mmlu/old")
    parser.add_argument("--save_dir", "-s", type=str, default="ftdata/mmlu")
    parser.add_argument("--concept_model_name", "-cn", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    main(args)
