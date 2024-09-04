import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    if include_answer:
        prompt += "\nAnswer:"
        prompt += " {}\n\n".format(df.iloc[idx, k + 1]) # answer
    else:
        prompt += "\nAnswer (choosing from A, B, C, and D):"
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

@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df, dev_expls_df, test_expls_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = "Now, you will answer the following question:\n"
        prompt_end += format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        if not pd.isna(test_expls_df.loc[i, "concepts"]) and not pd.isna(test_expls_df.loc[i, "explanations"]):
            expl_prompt = "Here are some concepts with explanations that may be useful:\n"
            expl_prompt += test_expls_df.loc[i, "explanations"]
            expl_prompt += "\n\n"
        elif not pd.isna(test_expls_df.loc[i, "concepts"]):
            expl_prompt = "Here are some concepts that may be useful:\n"
            expl_prompt += test_expls_df.loc[i, "concepts"]
            expl_prompt += "\n\n"
        else:
            expl_prompt = ""
        prompt = expl_prompt + train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = expl_prompt + train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        label = test_df.iloc[i, test_df.shape[1]-1]

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

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.6f} - {}".format(acc, subject))

    return cors, acc, all_probs


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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.exp_name))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.exp_name)))

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
                "expls_{}".format(args.expl_model_name),
                "dev",
                subject + "_expls.csv",
            )
        )
        test_expls_df = pd.read_csv(
            os.path.join(
                args.expl_dir,
                "expls_{}".format(args.expl_model_name),
                "test",
                subject + "_expls.csv",
            )
        )

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
    parser.add_argument("--expl_model_name", "-em", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    args.exp_name = f"{args.model_name}_feynman_{args.expl_model_name}_question_concept"
    main(args)
