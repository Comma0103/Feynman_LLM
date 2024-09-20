import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from tqdm import tqdm

choices = ["A", "B", "C", "D"]


def gen_expl_request(concept_with_taxo_path):
    request = (
        "Assume you are a teacher or an LLM trainer, tasked with teaching concepts of various disciplines.\n\n" + \
        "There is a concept(left part of ':') with its path in the taxonomy system(right part of ':'), related to some question:\n" + \
        "{0}\n" + \
        "where the 4 levels in the taxonomy path are 'discipline -> subject -> class session -> knowledge point' if the path is found, otherwise, understand it according to your knowledge.\n\n" + \
        "Based on your understanding of this concept at this taxonomy path (if found), " + \
        "could you briefly and clearly explain it to teach someone who is not familiar with it, " + \
        "or a smaller language model, so that they can understand it easily then answer the question it related to?\n\n" + \
        "Just simply explain the concept itself (NOT knowledge point) using the format 'concept: explanation' within 32 tokens, DO NOT use any font format or extra words:"
    ).format(concept_with_taxo_path)
    return request

def gen_ftdata(args, subject, dev_expls_df, val_expls_df, test_expls_df, expl_col_name, test_results_df, all_data_f=None):
    all_dialogs = {}
    df_names = ['dev', 'val', 'test']
    for df_idx, df in enumerate([dev_expls_df, val_expls_df, test_expls_df]):
        if df is None:
            continue
        print(df_names[df_idx], '..', sep='', end=' ')
        tic = time.time()
        
        dialogs = []
        for i in (range(df.shape[0])):
            if pd.isna(df.loc[i, "concepts"]) or pd.isna(df.loc[i, expl_col_name]):
                continue
            if test_results_df.loc[i, "{}_correct".format(args.model_name)] == False:
                continue
            concepts = df.loc[i, "concepts"].split(", ")
            if not pd.isna(df.loc[i, "taxonomy_path"]):
                concepts_with_taxo_path = df.loc[i, "taxonomy_path"].split(";|; ")
            else:
                concepts_with_taxo_path = ["Not found"] * len(concepts)
            explanations = df.loc[i, expl_col_name].split(";|; ")
            for concept, concept_with_taxo_path, expl in zip(concepts, concepts_with_taxo_path, explanations):
                if "Not found" in concept_with_taxo_path:
                    concept_with_taxo_path = concept + ": Not found"
                if "Not available" in expl:
                    continue
                # get prompt and make sure it fits
                request = gen_expl_request(concept_with_taxo_path)
                dialog = {
                    "messages": [
                        {"role": "system", "content": "You are a teacher or an LLM trainer."},
                        {"role": "user", "content": request},
                        {"role": "assistant", "content": expl}
                    ]
                }
                dialogs.append(dialog)
        
        all_dialogs[df_names[df_idx]] = dialogs
        
        toc = time.time()
        print("time: {:.3f}s".format(toc-tic), end='  ')
    
    return all_dialogs


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
    if not os.path.exists(os.path.join(args.save_dir, "ftdata_{}_sep_taxo_path_{}_correct_mix".format(args.expl_model_name, args.taxo_path_src))):
        os.makedirs(os.path.join(args.save_dir, "ftdata_{}_sep_taxo_path_{}_correct_mix".format(args.expl_model_name, args.taxo_path_src)))
    all_data_f = open(
        os.path.join(
            args.save_dir, 
            "ftdata_{}_sep_taxo_path_{}_correct_mix".format(args.expl_model_name, args.taxo_path_src), 
            "ftdata_{}_sep_taxo_path_{}_correct_mix.jsonl".format(args.expl_model_name, args.taxo_path_src)
        ), 
        "w", encoding="utf-8"
    )

    for subject in subjects:
        print('Generating finetune data for ', subject, '...', sep='', end='\t')
        tic = time.time()

        mixed_dialogs = {}
        iters_dict = {"":"explanations", "_2nd":"new_explanations", "_3rd":"third_explanations", "_4th":"fourth_explanations"}
        for iter_name, expl_col_name in iters_dict.items():
            dev_expls_df = None
            val_expls_df = None
            test_expls_df = pd.read_csv(
                os.path.join(
                    args.expl_dir,
                    "expls_{}_sep_taxo_path_{}{}".format(args.expl_model_name, args.taxo_path_src, iter_name),
                    "test",
                    subject + "{}_expls.csv".format(iter_name),
                )
            )
            
            test_results_df = pd.read_csv(
                os.path.join(
                    args.result_dir,
                    "results_{}_feynman_{}_sep_question_concept_taxo_path_{}{}".format(args.model_name, args.expl_model_name, args.taxo_path_src, iter_name),
                    subject + ".csv",
                )
            )

            iter_dialogs = gen_ftdata(args, subject, dev_expls_df, val_expls_df, test_expls_df, expl_col_name, test_results_df, all_data_f)
            
            for ds_split in iter_dialogs:
                if ds_split not in mixed_dialogs:
                    mixed_dialogs[ds_split] = []
                mixed_dialogs[ds_split].extend(iter_dialogs[ds_split])
        
        for ds_split in mixed_dialogs:
            os.makedirs(os.path.join(args.save_dir, "ftdata_{}_sep_taxo_path_{}_correct_mix".format(args.expl_model_name, args.taxo_path_src), ds_split), exist_ok=True)
            data_file = os.path.join(
                args.save_dir, "ftdata_{}_sep_taxo_path_{}_correct_mix".format(args.expl_model_name, args.taxo_path_src), ds_split, subject + "_ftdata.jsonl"
            )
            with open(data_file, 'w', encoding="utf-8") as f:
                for dialog in mixed_dialogs[ds_split]:
                    json_record = json.dumps(dialog)
                    f.write(json_record + "\n")
                    if all_data_f:
                        all_data_f.write(json_record + "\n")
        
        toc = time.time()
        print("\nTime: {:.3f} s, {} of {}\n".format(toc-tic, subjects.index(subject)+1, len(subjects)))
    
    all_data_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/mmlu_data")
    parser.add_argument("--expl_dir", "-e", type=str, default="explanations/mmlu")
    parser.add_argument("--result_dir", "-r", type=str, default="results/mmlu")
    parser.add_argument("--save_dir", "-s", type=str, default="ftdata/mmlu")
    parser.add_argument("--model_name", "-n", type=str, default="Meta-Llama-2-7B")
    parser.add_argument("--taxo_path_src", "-tp", type=str, default="gen", choices=["gen", "search"])
    parser.add_argument("--expl_model_name", "-en", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    main(args)