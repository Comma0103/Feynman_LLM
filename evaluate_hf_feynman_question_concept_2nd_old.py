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

def gen_2nd_concept_request(df, subject, i, old_concept_expl, stu_pred):
    q_with_a = format_example(df, i)
    request = (
        "Assume you are a teacher or an LLM trainer, tasked with teaching about {0}.\n\n" + \
        "There is a multiple choice question (with answer) about {0}:\n" + \
        "{1}" + \
        "You used to give the concepts related to this question with their explanation as follow:\n" + \
        "{2}\n\n" + \
        "However, a student LM got a wrong answer ({3}) after adding your explanation into its prompt when answering this question.\n\n" + \
        "Based on your understanding of this question and {0}, think about the reason why the student got this wrong answer, " + \
        "could you modify the explanation of these concepts to teach someone who is not familiar with {0}, " + \
        "or a smaller language model, so that they can answer this question right? " + \
        "Just simply list and explain the concepts within 256 tokens, DO NOT include the answer or use any font format:"
    ).format(format_subject(subject), q_with_a, old_concept_expl, stu_pred)
    return request

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

def eval(args, subject, model, tokenizer, dev_df, test_df, dev_concepts_df, test_concepts_df, teacher_client):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]
    new_concepts = []

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        concept_prompt = test_concepts_df.loc[i, "concepts"] if not pd.isna(test_concepts_df.loc[i, "concepts"]) else ""
        prompt = concept_prompt + train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = concept_prompt + train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        label = test_df.iloc[i, test_df.shape[1]-1]

        probs, pred = forward(args, model, tokenizer, input_ids)
        cor = pred == label
        
        if not cor:
            concept_prompt = teacher_client.call(gen_2nd_concept_request(test_df, subject, i, concept_prompt, pred))
            concept_prompt = "" if concept_prompt is None else concept_prompt
            prompt = concept_prompt + train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            probs, pred = forward(args, model, tokenizer, input_ids)
            cor = pred == label
        new_concept = concept_prompt if not cor and concept_prompt != "" else None
        new_concepts.append(new_concept)
        
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.6f} - {}".format(acc, subject))

    return cors, acc, all_probs, new_concepts


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
    
    oai_client = Openai(
        apis=API_INFOS[args.concept_model_name]
    )
    # res = oai_client.call("hello")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.exp_name))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.exp_name)))
    if not os.path.exists(os.path.join(args.concept_dir, "concepts_{}_2nd".format(args.concept_model_name), "test")):
        os.makedirs(os.path.join(args.concept_dir, "concepts_{}_2nd".format(args.concept_model_name), "test"))

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
        dev_concepts_df = pd.read_csv(
            os.path.join(
                args.concept_dir,
                "concepts_{}".format(args.concept_model_name),
                "dev",
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

        cors, acc, probs, new_concepts = eval(args, subject, model, tokenizer, dev_df, test_df, dev_concepts_df, test_concepts_df, oai_client)
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
        test_concepts_df["new_concepts"] = new_concepts
        test_concepts_df.to_csv(
            os.path.join(
                args.concept_dir, "concepts_{}_2nd".format(args.concept_model_name), "test", "{}_2nd_concepts.csv".format(subject)
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

    results_file = os.path.join(
        args.save_dir, "accuracies_{}.json".format(args.exp_name)
    )
    with open(results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/mmlu_data")
    parser.add_argument("--concept_dir", "-c", type=str, default="concepts")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model_path", "-m", type=str, default="/home/lidong1/qilongma/blob/public_models/Meta-Llama-3-8B")
    parser.add_argument("--model_name", "-n", type=str, default="Meta-Llama-3-8B")
    parser.add_argument("--concept_model_name", "-cm", type=str, default="OpenAI-GPT-4o-mini")
    args = parser.parse_args()
    args.exp_name = f"{args.model_name}_feynman_{args.concept_model_name}_question_concept_2nd"
    main(args)
