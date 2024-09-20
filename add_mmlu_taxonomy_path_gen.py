import argparse
import time
import os
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor

from call_gpt import Openai, API_INFOS

from functools import partial

from tqdm import tqdm

choices = ["A", "B", "C", "D"]

teacher_client = Openai(
    apis=API_INFOS['OpenAI-GPT-4o-mini']
)

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0] # question
    k = df.shape[1] - 3 # no. of choices
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1]) # choices
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1]) # answer
    return prompt

def gen_taxonomy_path_request(mmlu_subject, df, i, concept):
    subject = mmlu_subject.replace("_", " ")
    q_with_a = format_example(df, i)
    request = (
        "Assume you are a teacher or an LLM trainer, tasked with teaching about {0}.\n\n" + \
        "There is a multiple choice question (with answer) about {0}:\n" + \
        "{1}" + \
        "You used to extract a concept related to this question: {2}.\n\n" + \
        "Based on your understanding of this question and {2} in {0}, " + \
        "could you give an accurate path of {2} in {0} without ambiguity in the taxonomy system for this question, to teach someone who is not familiar with {0}, " + \
        "or a smaller language model, to help them answer this question better?\n\n" + \
        "Concretely, the taxonomy path of a concept has 4 levels, i.e., discipline -> subject -> class session -> knowledge point. " + \
        "The concept can be the knowledge point itself or contained in it. " + \
        "Most crucially, the words employed at each path level within the taxonomy system should be as consistent or similar as feasible for synonyms or comparable concepts in diverse questions, i.e., do not create too many disciplines or subjects.\n\n" + \
        "Here are some examples in the format of \'concept: taxonomy path\' :\n" + \
        "Cyclic Subgroup: Mathematics -> Abstract Algebra -> Group Theory -> Subgroup (for a question in Abstract Algebra)\n" + \
        "Newton's third law: Physics -> Classical Mechanics -> Newton's laws -> Newton's third law (for a question in High School Physics)\n" + \
        "Chloroplast: Biology -> Cell and Molecular Biology -> Cell Structure -> Mitochondria and Chloroplasts (for a question in College Biology)\n" + \
        "Function: Computer Science -> Discrete Mathematics -> Functions and Relations -> Function (for a question in College Computer Science)\n" + \
        "Italian Unification: European History -> Late Modern Period -> Post-Napoleonic Europe and the Rise of Nationalism -> German and Italian Unification\n\n"
        "Just simply respond the taxonomy path of the given concept (levels separated by -> and space, uppercase for the first letter, use the same format as the examples above) only within 128 tokens, " + \
        "DO NOT include other content or use any font format:" # 不同题目，所建体系中各级划分要统一！！ DO NOT number or explain them, and  or use any font format
    ).format(subject, q_with_a, concept)
    return request

def process_mmlu_file(file_path, output_split_dir):#, teacher_client):
    try:
        cnt_404_file, cnt_tot_file = 0, 0
        
        tic = time.time()
        
        df = pd.read_csv(file_path)
        mmlu_subject = os.path.basename(file_path).split('_concepts.csv')[0]

        taxonomy_paths = []
        for i in (range(len(df))):
            if pd.isna(df.loc[i, "concepts"]):
                taxonomy_paths.append(None)
            else:
                concepts = df.loc[i, "concepts"].split(", ")
                paths = []
                for concept in concepts:
                    path = teacher_client.call(gen_taxonomy_path_request(mmlu_subject, df, i, concept))
                    paths.append(path if path is not None else concept + ": Not found")
                    cnt_tot_file += 1
                    if path is None:
                        cnt_404_file += 1
                taxonomy_paths.append(";|; ".join(paths))

        df['taxonomy_path'] = taxonomy_paths

        # Save to new location
        os.makedirs(output_split_dir, exist_ok=True)
        output_file_path = os.path.join(output_split_dir, os.path.basename(file_path))
        df.to_csv(output_file_path, index=False)

        toc = time.time()
        split = os.path.basename(os.path.dirname(file_path))
        print(f"Processed {split}/{os.path.basename(file_path)} in {toc - tic:.3f} seconds")
        print(f"File - 404: {cnt_404_file}, total: {cnt_tot_file}, 404 rate: {cnt_404_file/cnt_tot_file}")
        return cnt_404_file, cnt_tot_file
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0

def process_mmlu_files(args):
    # Initialize teacher client
    oai_client = Openai(
        apis=API_INFOS[args.concept_model_name]
    )
    # res = oai_client.call("hello")
    # fn = partial(process_mmlu_file, teacher_client=oai_client)
    fn = process_mmlu_file
    
    # Collect all file paths
    file_paths = [[], []]
    for split in ['dev', 'val', 'test']:
        split_dir = os.path.join(args.concept_dir, split)
        for file in sorted(os.listdir(split_dir)):
            if file.endswith('_concepts.csv'):
                file_paths[0].append(os.path.join(split_dir, file))
                file_paths[1].append(os.path.join(args.output_dir, split))
    print(f"Found {len(file_paths[0])} files to process")\
    
    # Use multiprocessing to process files in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(fn, file_paths[0], file_paths[1]))
    
    cnt_404, cnt_tot = map(sum, zip(*results))
    print(f"\nFinal - 404: {cnt_404}, total: {cnt_tot}, 404 rate: {cnt_tot and cnt_404 / cnt_tot or 'N/A'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument('--concept_dir', type=str, 
                        default='/home/lidong1/qilongma/Feynman_LLM/concepts/mmlu/concepts_OpenAI-GPT-4o-mini', 
                        help='Directory containing MMLU dataset concepts')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/lidong1/qilongma/Feynman_LLM/concepts/mmlu/concepts_OpenAI-GPT-4o-mini_taxo_path_gen', 
                        help='Output directory for generated taxonomy paths')
    parser.add_argument("--concept_model_name", "-cn", type=str, 
                        default="OpenAI-GPT-4o-mini", 
                        help='Concept model name to use for generating taxonomy paths')
    parser.add_argument('--max_workers', type=int, 
                        default=8, 
                        help='Number of workers to use for processing files in parallel')
    args = parser.parse_args()
    
    # Run the process
    process_mmlu_files(args)
