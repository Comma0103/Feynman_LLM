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

# TODO: under construction


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

def gen_expl(args, subject, teacher_client, dev_concepts_df, val_concepts_df, test_concepts_df):
    df_names = ['dev', 'val', 'test']
    for df_idx, df in enumerate([dev_concepts_df, val_concepts_df, test_concepts_df]):
        save_path = os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}".format(args.expl_model_name, args.taxo_path_src), df_names[df_idx], subject + "_expls.csv")
        if os.path.exists(save_path):
            print(df_names[df_idx], ' already exists, skipping...', sep='', end=' ')
            continue
        
        print(df_names[df_idx], '..', sep='', end='\n')
        tic = time.time()
        
        all_expls = []
        for i in tqdm(range(df.shape[0])):
            if pd.isna(df.loc[i, "taxonomy_path"]):
                all_expls.append(None)
                continue
            concepts = df.loc[i, "concepts"].split(", ")
            concepts_with_taxo_path = df.loc[i, "taxonomy_path"].split(";|; ")
            expls = []
            for concept, concept_with_taxo_path in zip(concepts, concepts_with_taxo_path):
                if "Not found" in concept_with_taxo_path:
                    concept_with_taxo_path = concept + ": Not found"
                # get prompt and make sure it fits
                expl = teacher_client.call(gen_expl_request(concept_with_taxo_path)) 
                expls.append(expl if expl is not None else concept + ": Not available")
            all_expls.append(";|; ".join(expls))
        df['explanations'] = all_expls
        
        os.makedirs(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}".format(args.expl_model_name, args.taxo_path_src), df_names[df_idx]), exist_ok=True)
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
        apis=API_INFOS[args.ins_model_name]
    )
    # res = oai_client.call("hello")

    if not os.path.exists(args.ins_dir):
        os.makedirs(args.ins_dir)
    # TODO: ================= resume from here =================
    if not os.path.exists(os.path.join(args.ins_dir, "ins_{}_sep_taxo_path_{}".format(args.ins_model_name, args.taxo_path_src))):
        os.makedirs(os.path.join(args.ins_dir, "ins_{}_sep_taxo_path_{}".format(args.ins_model_name, args.taxo_path_src)))

    for subject in subjects:
        print('Generating explanations for ', subject, '...', sep='', end='\t')
        tic = time.time()
        
        dev_concepts_df = pd.read_csv(
            os.path.join(
                args.concept_dir,
                "concepts_{}_taxo_path_{}".format(args.concept_model_name, args.taxo_path_src),
                "dev",
                subject + "_concepts.csv",
            )
        )
        val_concepts_df = pd.read_csv(
            os.path.join(
                args.concept_dir,
                "concepts_{}_taxo_path_{}".format(args.concept_model_name, args.taxo_path_src),
                "val",
                subject + "_concepts.csv",
            )
        )
        test_concepts_df = pd.read_csv(
            os.path.join(
                args.concept_dir,
                "concepts_{}_taxo_path_{}".format(args.concept_model_name, args.taxo_path_src),
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
    parser.add_argument("--expl_dir", "-e", type=str, default="explanations")
    parser.add_argument("--ins_dir", "-i", type=str, default="instances")
    # parser.add_argument("--expl_model_path", "-em", type=str, default="/home/lidong1/qilongma/blob/public_models/xxx")
    parser.add_argument("--expl_model_name", "-en", type=str, default="OpenAI-GPT-4o-mini")
    parser.add_argument("--taxo_path_src", "-tp", type=str, default="gen", choices=["gen", "search"])
    # parser.add_argument("--ins_model_path", "-im", type=str, default="/home/lidong1/qilongma/blob/public_models/xxx")
    parser.add_argument("--ins_model_name", "-in", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    main(args)
