import argparse
import os
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from call_gpt import Openai, API_INFOS  # 确保这个模块可用

from tqdm import tqdm

sub_folders = ["dev", "val", "test"]

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = " ".join(l)
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0] # question
    k = df.shape[1] - 3 # no. of choices
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
        "could you briefly and clearly extract few central concepts covered in the question to teach someone who is not familiar with {0}, " + \
        "or a smaller language model, so that they can answer this question? " + \
        "Just simply list the extracted concepts (separated by comma and space, uppercase for the first letter) themselves only within 64 tokens, " + \
        "DO NOT include the answer:" # DO NOT number or explain them, and  or use any font format
    ).format(format_subject(subject), q_with_a)
    return request

def fix_concepts(args, subject, teacher_client, df, df_name):
    print(df_name, '..', sep='', end=' ')
    tic = time.time()
    
    concepts = []
    # use tqdm to show progress
    for i in tqdm(range(df.shape[0])):
        if pd.isna(df.loc[i, "concepts"]):
            # 生成概念并添加到列表中
            concept = teacher_client.call(gen_concept_request(df, subject, i))
        else:
            concept = df.loc[i, "concepts"]
        concepts.append(concept)
    df['concepts'] = concepts
    
    # 保存生成后的文件
    os.makedirs(os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name), df_name), exist_ok=True)
    df.to_csv(
        os.path.join(
            args.concept_dir, "concepts_{}".format(args.concept_model_name), df_name, subject + "_concepts.csv"
        ),
        index=False,
    )
    
    toc = time.time()
    print("time: {:.3f}s".format(toc-tic))


def check_and_fix_concepts(args, subject, df_name, df, teacher_client):
    # 检查 "concepts" 列是否缺少元素
    nan_questions = df["questions"].isna().sum()
    nan_concepts = df["concepts"].isna().sum()
    len_questions = len(df["questions"])
    len_concepts = len(df["concepts"])
    if nan_questions != nan_concepts or len_questions != len_concepts:
        print(f"File {df_name} / {subject}_concepts.csv has inconsistent 'questions'(len: {len_questions}, nan: {nan_questions}) and 'concepts'(len: {len_concepts}, nan: {nan_concepts}) columns.")
        if args.fix:
            print(f"Regenerating concepts...")
            fix_concepts(args, subject, teacher_client, df, df_name)
    else:
        pass
        # print(f"File {subject}_{df_name}.csv is consistent.")


def main(args):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    
    oai_client = Openai(apis=API_INFOS[args.concept_model_name])

    if not os.path.exists(args.concept_dir):
        os.makedirs(args.concept_dir)
    if not os.path.exists(os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name))):
        os.makedirs(os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name)))

    for subject in subjects:
        # print('Checking concepts for ', subject, '...', sep='', end='\t')
        
        for sub_folder in sub_folders:
            df_path = os.path.join(args.concept_dir, "concepts_{}".format(args.concept_model_name), sub_folder, subject + f"_concepts.csv")
            df = pd.read_csv(df_path)
            # print(df.columns == ["questions", "A", "B", "C", "D", "answers", "concepts"])

            check_and_fix_concepts(args, subject, sub_folder, df, oai_client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="/data/qilongma/mmlu_data")
    parser.add_argument("--concept_dir", "-c", type=str, default="concepts/mmlu")
    parser.add_argument("--concept_model_name", "-cn", type=str, default="OpenAI-GPT-4o-mini")
    parser.add_argument("--fix", "-f", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
