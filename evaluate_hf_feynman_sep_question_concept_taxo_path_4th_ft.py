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
    return subject.replace("_", " ")

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0] # question
    k = df.shape[1] - 2 # no. of choices
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1]) # choices
    if include_answer:
        prompt += "\nAnswer:"
        prompt += " {}\n\n".format(df.iloc[idx, k + 1]) # answer
    else:
        prompt += "\nAnswer (Just choosing from A, B, C, and D without any extra words):"
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0] # no. of questions
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

# def gen_4th_expl_request(concept_with_taxo_path, third_concept_expl):
#     request = (
#         "Assume you are a teacher or an LLM trainer, tasked with teaching concepts of various disciplines.\n\n" + \
#         "There is a concept(left part of ':') with its path in the taxonomy system(right part of ':'), related to some question:\n" + \
#         "{0}\n" + \
#         "where the 4 levels in the taxonomy path are 'discipline -> subject -> class session -> knowledge point' if the path is found, otherwise, understand it according to your knowledge.\n\n" + \
#         "You used to give the explanation of this concept as follow:\n" + \
#         "{1}\n\n" + \
#         "However, a student LM got a wrong answer after adding your explanation into its prompt when answering the question.\n\n" + \
#         "Based on your understanding of this concept, think about the reason why the student got the wrong answer, " + \
#         "could you modify the explanation of this concept to teach someone who is not familiar with it, " + \
#         "or a smaller language model, so that they can understand it easily then answer the question right?\n\n" + \
#         "Just simply explain the concept itself (NOT knowledge point) using the format 'concept: explanation' within 32 tokens, DO NOT use any font format or extra words:"
#     ).format(concept_with_taxo_path, third_concept_expl)
#     return request

def gen_expl_prompt(concepts=None, taxonomy_paths=None, explanations=None):
    if not pd.isna(concepts) and not pd.isna(taxonomy_paths) and not pd.isna(explanations):
        expl_prompt = "Here are some concepts with their path in the taxonomy system (if not found, understand according to your knowledge), " + \
                "and explanations of the concepts (if not available, understand according to your knowledge), which may be useful:\n"
        concepts_list = concepts.split(", ")
        taxonomy_paths_list = [e.split(": ")[1] if ": " in e else e for e in taxonomy_paths.split(";|; ")]
        explanations_list = [e.split(": ")[1] if ": " in e else e for e in explanations.split(";|; ")]
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
def forward(args, model, tokenizer, input_ids):
    logits = model(input_ids=input_ids).logits[0, -1]

    probs = (
        torch.nn.functional.softmax(
            torch.tensor(
                [
                    logits[tokenizer("A").input_ids[-1]],
                    logits[tokenizer("B").input_ids[-1]],
                    logits[tokenizer("C").input_ids[-1]],
                    logits[tokenizer("D").input_ids[-1]],
                ]
            ).float(),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    return probs, pred

def eval(args, subject, model, tokenizer, dev_df, test_df, dev_expls_df, test_expls_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]
    # all_fourth_expls = []

    for i in tqdm(range(test_df.shape[0]), ncols=75):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = "Now, you will answer the following question:\n"
        prompt_end += format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        expl_prompt = gen_expl_prompt(
            test_expls_df.loc[i, "concepts"],
            test_expls_df.loc[i, "taxonomy_path"],
            test_expls_df.loc[i, "explanations"]
        )
        prompt = expl_prompt + train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = expl_prompt + train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        label = test_df.iloc[i, test_df.shape[1]-1]

        probs, pred = forward(args, model, tokenizer, input_ids)
        cor = pred == label
        
        # if not cor:
        #     if pd.isna(test_expls_df.loc[i, "taxonomy_path"]):
        #         fourth_expls = None
        #         all_fourth_expls.append(fourth_expls)
        #     else:
        #         concepts = test_expls_df.loc[i, "concepts"].split(", ")
        #         concepts_with_taxo_path = test_expls_df.loc[i, "taxonomy_path"].split(";|; ")
        #         if pd.isna(test_expls_df.loc[i, "third_explanations"]):
        #             third_expls = [concept + ": Not available" for concept in concepts]
        #         else:
        #             third_expls = test_expls_df.loc[i, "third_explanations"].split(";|; ")
        #         fourth_expls = []
        #         for concept, concept_with_taxo_path, third_expl in zip(concepts, concepts_with_taxo_path, third_expls):
        #             if "Not found" in concept_with_taxo_path:
        #                 concept_with_taxo_path = concept + ": Not found"
        #             # get prompt and make sure it fits
        #             fourth_expl = teacher_client.call(gen_4th_expl_request(concept_with_taxo_path, third_expl)) 
        #             fourth_expls.append(fourth_expl if fourth_expl is not None else concept + ": Not available")
        #         all_fourth_expls.append(";|; ".join(fourth_expls))
            
        #     expl_prompt = gen_expl_prompt(
        #         test_expls_df.loc[i, "concepts"],
        #         test_expls_df.loc[i, "taxonomy_path"],
        #         ";|; ".join(fourth_expls) if fourth_expls is not None else None,
        #         ";|; ".join(third_expls) if third_expls is not None else None,
        #     )
        #     prompt = expl_prompt + train_prompt + prompt_end
        #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        #     probs, pred = forward(args, model, tokenizer, input_ids)
        #     cor = pred == label
        # else:
        #     all_fourth_expls.append(None)
        
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.6f} - {}".format(acc, subject))

    return cors, acc, all_probs#, all_fourth_expls


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
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    
    # explanations have been generated in expls_OpenAI-GPT-4o-mini-0903-gen-4th_sep_taxo_path_gen
    # oai_client = Openai(
    #     apis=API_INFOS[args.expl_model_name]
    # )
    # res = oai_client.call("hello")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.exp_name))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.exp_name)))
    # if not os.path.exists(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}_4th_ft".format(args.expl_model_name, args.taxo_path_src), "test")):
    #     os.makedirs(os.path.join(args.expl_dir, "expls_{}_sep_taxo_path_{}_4th_ft".format(args.expl_model_name, args.taxo_path_src), "test"))

    all_cors = []
    subject_cors = {}
    subcat_cors = {
        subcat: [] for subcat_list in subcategories.values() for subcat in subcat_list
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        tic = time.time()
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        dev_expls_df = pd.read_csv(
            os.path.join(
                args.expl_dir,
                "expls_{}_sep_taxo_path_{}".format(args.expl_model_name, args.taxo_path_src),
                "dev",
                subject + "_expls.csv",
            )
        )
        test_expls_df = pd.read_csv(
            os.path.join(
                args.expl_dir,
                "expls_{}_sep_taxo_path_{}".format(args.expl_model_name, args.taxo_path_src),
                "test",
                subject + "_expls.csv",
            )
        )

        # cors, acc, probs, all_fourth_expls = eval(args, subject, model, tokenizer, dev_df, test_df, dev_expls_df, test_expls_df, oai_client)
        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df, dev_expls_df, test_expls_df)
        subject_cors[subject] = cors.tolist()
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model_name)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model_name, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.exp_name), "{}.csv".format(subject)
            ),
            index=None,
        )
        # test_expls_df["fourth_explanations"] = all_fourth_expls
        # test_expls_df.to_csv(
        #     os.path.join(
        #         args.expl_dir, "expls_{}_sep_taxo_path_{}_4th_ft".format(args.expl_model_name, args.taxo_path_src), "test", "{}_4th_expls.csv".format(subject)
        #     ),
        #     index=None,
        # )
        toc = time.time()
        print("\tTime: {:.3f} s, {} of {}\n".format(toc-tic, subjects.index(subject)+1, len(subjects)))

    results = {"subcategories": {}, "categories": {}, "subjects": {}}
    print("Subject accuracies:")
    for subject in subject_cors:
        subject_acc = np.mean(subject_cors[subject])
        results["subjects"][subject] = subject_acc
        print("Average accuracy {:.3f} - {}".format(subject_acc, subject))
    print("\nSubcategory accuracies:")
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
    print("\nCategory accuracies:")
    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
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
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/mmlu_data")
    parser.add_argument("--expl_dir", "-e", type=str, default="explanations")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/Meta-Llama-2-7B-hf")
    parser.add_argument("--model_name", "-n", type=str, default="Meta-Llama-2-7B")
    parser.add_argument("--taxo_path_src", "-tp", type=str, default="gen", choices=["gen", "search"])
    parser.add_argument("--expl_model_name", "-en", type=str, default="OpenAI-GPT-4o-mini-0903-gen-4th")
    args = parser.parse_args()
    args.exp_name = f"{args.model_name}_feynman_{args.expl_model_name}_sep_question_concept_taxo_path_{args.taxo_path_src}_just"
    main(args)
