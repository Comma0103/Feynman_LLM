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


def gen_expl_request(concept_with_taxo_path):
    request = (
        "Assume you are a teacher or an LLM trainer, tasked with teaching concepts of various disciplines about science exam questions that span several grades.\n\n" + \
        "There is a concept(left part of ':') with its path in the human knoeledge taxonomy system(right part of ':'):\n\n" + \
        "{0}\n\n" + \
        "where the 4 levels in the taxonomy path are 'discipline -> subject -> class session -> knowledge point' if the path is found, otherwise, understand it according to your knowledge.\n\n" + \
        "Based on your understanding of this concept at this taxonomy path (if found), " + \
        "could you briefly and clearly explain it to teach someone who is not familiar with it, " + \
        "or a smaller language model, so that they can understand it easily then answer the question it related to?\n\n" + \
        "Just simply explain the concept itself (NOT knowledge point) using the format 'concept: explanation' within 32 tokens, DO NOT use any font format or extra words:"
    ).format(concept_with_taxo_path)
    return request

def gen_expl(args, level, teacher_client, dev_concept_data_list, train_concept_data_list, test_concept_data_list):
    split_names = ['Dev', 'Train', 'Test']
    for idx, concept_data_list in enumerate([dev_concept_data_list, train_concept_data_list, test_concept_data_list]):
        save_path = os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}".format(args.expl_model_name, args.taxo_path_src), f"{level}-{split_names[idx]}_expls.jsonl")
        if os.path.exists(save_path):
            print(split_names[idx], ' already exists, skipping...', sep='', end='\n')
            continue
        
        print(split_names[idx], '..', sep='', end='\n')
        tic = time.time()
        
        for i in tqdm(range(len(concept_data_list)), ncols=75):
            if pd.isna(concept_data_list[i]["taxonomy_path"]) or concept_data_list[i]["taxonomy_path"] == "":
                concept_data_list[i]['explanations'] = None
                continue
            concepts = concept_data_list[i]["concepts"].split(", ")
            concepts_with_taxo_path = concept_data_list[i]["taxonomy_path"].split(";|; ")
            expls = []
            for concept, concept_with_taxo_path in zip(concepts, concepts_with_taxo_path):
                if "Not found" in concept_with_taxo_path:
                    concept_with_taxo_path = concept + ": Not found"
                # get prompt and make sure it fits
                expl = teacher_client.call(gen_expl_request(concept_with_taxo_path)) 
                expls.append(expl if expl is not None else concept + ": Not available")
            concept_data_list[i]['explanations'] = ";|; ".join(expls)
        
        os.makedirs(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}".format(args.expl_model_name, args.taxo_path_src)), exist_ok=True)
        with open(save_path, 'w', encoding="utf-8") as out_f:
            for concept_data in concept_data_list:
                out_f.write(json.dumps(concept_data) + "\n")
        
        toc = time.time()
        print("time: {:.3f}s".format(toc-tic), end='\n')

def main(args):
    difficulty_levels = sorted(
        [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    ) # ['ARC-Challenge', 'ARC-Easy']
    
    oai_client = Openai(
        apis=API_INFOS[args.expl_model_name]
    )
    # res = oai_client.call("hello")

    if not os.path.exists(args.expl_dir):
        os.makedirs(args.expl_dir)
    if not os.path.exists(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}".format(args.expl_model_name, args.taxo_path_src))):
        os.makedirs(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}".format(args.expl_model_name, args.taxo_path_src)))

    for level in difficulty_levels:
        print('Generating explanations for ', f'{level}', '...', sep='', end='\n')
        tic = time.time()
        
        model_concept_dir = os.path.join(args.concept_dir, "concepts_{}_taxo_path_{}".format(args.concept_model_name, args.taxo_path_src))
        with open(os.path.join(model_concept_dir, f'{level}-Dev_concepts.jsonl'), 'r', encoding='utf-8') as dev_concept_f:
            dev_concept_data_list = [json.loads(line) for line in dev_concept_f]
        with open(os.path.join(model_concept_dir, f'{level}-Train_concepts.jsonl'), 'r', encoding='utf-8') as train_concept_f:
            train_concept_data_list = [json.loads(line) for line in train_concept_f]
        with open(os.path.join(model_concept_dir, f'{level}-Test_concepts.jsonl'), 'r', encoding='utf-8') as test_concept_f:
            test_concept_data_list = [json.loads(line) for line in test_concept_f]

        gen_expl(args, level, oai_client, dev_concept_data_list, train_concept_data_list, test_concept_data_list)
        
        toc = time.time()
        print("\nTime: {:.3f} s, {} done\n".format(toc-tic, level))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/ARC-V1-Feb2018")
    parser.add_argument("--concept_dir", "-c", type=str, default="concepts/arc")
    parser.add_argument("--expl_dir", "-e", type=str, default="explanations/arc")
    # parser.add_argument("--concept_model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/xxx")
    parser.add_argument("--concept_model_name", "-cn", type=str, default="OpenAI-GPT-4o-mini")
    parser.add_argument("--taxo_path_src", "-tp", type=str, default="gen", choices=["gen", "search"])
    # parser.add_argument("--expl_model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/xxx")
    parser.add_argument("--expl_model_name", "-en", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    main(args)
