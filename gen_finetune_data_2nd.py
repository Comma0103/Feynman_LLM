import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from tqdm import tqdm

choices = ["A", "B", "C", "D"]


def gen_expl_request(df, i):
    concepts = df.loc[i, "concepts"]
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

def gen_ftdata(args, subject, dev_expls_df, val_expls_df, test_expls_df, all_data_f=None):
    df_names = ['dev', 'val', 'test']
    for df_idx, df in enumerate([dev_expls_df, val_expls_df, test_expls_df]):
        if df is None:
            continue
        print(df_names[df_idx], '..', sep='', end=' ')
        tic = time.time()
        
        dialogs = []
        for i in (range(df.shape[0])):
            if pd.isna(df.loc[i, "concepts"]) or pd.isna(df.loc[i, "new_explanations"]):
                continue
            request = gen_expl_request(df, i)
            expl = df.loc[i, "new_explanations"]
            dialog = {
                "messages": [
                    {"role": "system", "content": "You are a teacher or an LLM trainer."},
                    {"role": "user", "content": request},
                    {"role": "assistant", "content": expl}
                ]
            }
            dialogs.append(dialog)
        
        os.makedirs(os.path.join(args.save_dir, "ftdata_{}_2nd".format(args.expl_model_name), df_names[df_idx]), exist_ok=True)
        data_file = os.path.join(
            args.save_dir, "ftdata_{}_2nd".format(args.expl_model_name), df_names[df_idx], subject + "_ftdata.jsonl"
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
    if not os.path.exists(os.path.join(args.save_dir, "ftdata_{}_2nd".format(args.expl_model_name))):
        os.makedirs(os.path.join(args.save_dir, "ftdata_{}_2nd".format(args.expl_model_name)))
    all_data_f = open(os.path.join(
        args.save_dir, "ftdata_{}_2nd".format(args.expl_model_name), "ftdata_{}_2nd.jsonl".format(args.expl_model_name)), "w", encoding="utf-8")

    for subject in subjects:
        print('Generating finetune data for ', subject, '...', sep='', end='\t')
        tic = time.time()

        dev_expls_df = None
        val_expls_df = None
        test_expls_df = pd.read_csv(
            os.path.join(
                args.expl_dir,
                "expls_{}_2nd".format(args.expl_model_name),
                "test",
                subject + "_2nd_expls.csv",
            )
        )

        gen_ftdata(args, subject, dev_expls_df, val_expls_df, test_expls_df, all_data_f)
        
        toc = time.time()
        print("\nTime: {:.3f} s, {} of {}\n".format(toc-tic, subjects.index(subject)+1, len(subjects)))
    
    all_data_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/mmlu_data")
    parser.add_argument("--expl_dir", "-e", type=str, default="explanations")
    parser.add_argument("--save_dir", "-s", type=str, default="ftdata")
    parser.add_argument("--expl_model_name", "-em", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    main(args)