import argparse
import time
import os
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor

from call_gpt import Openai, API_INFOS

from functools import partial

from tqdm import tqdm


teacher_client = Openai(
    apis=API_INFOS['OpenAI-GPT-4o-mini']
)

def format_example(question, choices, answer, include_answer=True):
    choices = sorted(choices, key=lambda x: x['label'])
    prompt = question # question
    k = len(choices) # no. of choices
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j]['label'], choices[j]['text']) # choices
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(answer) # answer
    return prompt

def gen_taxonomy_path_request(question, choices, answer, level, grade, concept):
    q_with_a = format_example(question, choices, answer)
    request = (
        "Assume you are a teacher or an LLM trainer, tasked with teaching about science exam questions that span several grades. " + \
        "Each question has a multiple choice structure (typically 4 answer options, some could have 3 or 5). " + \
        "The questions are sorted into a Challenge Set of 'hard' questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of questions.\n\n" + \
        "There is a multiple choice question (with answer) at grade {1} in the {0} set:\n\n" + \
        "{2}" + \
        "You used to extract a concept related to this question: {3}.\n\n" + \
        "Based on the information above and your understanding of this question and {3} taught in grade {1}, " + \
        "could you give an accurate path of {3} without ambiguity in the human knowledge taxonomy system for this question, to teach someone who is not familiar with the relative knowledges, " + \
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
    ).format(level, grade, q_with_a, concept)
    return request

def process_arc_file(file_path, output_dir):#, teacher_client):
    try:
        cnt_404_file, cnt_tot_file = 0, 0
        
        tic = time.time()
        
        with open(file_path, 'r', encoding='utf-8') as concept_f:
            concept_data_list = [json.loads(line) for line in concept_f]

        for i in tqdm(range(len(concept_data_list)), ncols=75):
            concepts = concept_data_list[i]["concepts"]
            if pd.isna(concepts) or concepts == "":
                concept_data_list[i]['taxonomy_path'] = None
            else:
                question = concept_data_list[i]['question']['stem']
                choices = concept_data_list[i]['question']['choices']
                answer = concept_data_list[i]['answerKey']
                level = 'challenge' if 'Challenge' in os.path.basename(file_path) else 'easy' if 'Easy' in os.path.basename(file_path) else 'unknown'
                grade = concept_data_list[i]['grade']
                concepts_list = concepts.split(", ")
                paths = []
                for concept in concepts_list:
                    path = teacher_client.call(gen_taxonomy_path_request(question, choices, answer, level, grade, concept))
                    paths.append(path if path is not None else concept + ": Not found")
                    cnt_tot_file += 1
                    if path is None:
                        cnt_404_file += 1
                concept_data_list[i]['taxonomy_path'] = ";|; ".join(paths)

        # Save to new location
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(output_file_path, 'w', encoding="utf-8") as out_f:
            for concept_data in concept_data_list:
                out_f.write(json.dumps(concept_data) + "\n")

        toc = time.time()
        print(f"Processed {os.path.basename(file_path)} in {toc - tic:.3f} seconds")
        print(f"File - 404: {cnt_404_file}, total: {cnt_tot_file}, 404 rate: {cnt_404_file/cnt_tot_file}")
        return cnt_404_file, cnt_tot_file
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0

def process_arc_files(args):
    # Initialize teacher client
    oai_client = Openai(
        apis=API_INFOS[args.concept_model_name]
    )
    # res = oai_client.call("hello")
    # fn = partial(process_arc_file, teacher_client=oai_client)
    fn = process_arc_file
    
    # Collect all file paths
    file_paths = [[], []]
    for file in sorted(os.listdir(args.concept_dir)):
        if file.startswith('ARC-') and file.endswith('_concepts.jsonl'):
            file_paths[0].append(os.path.join(args.concept_dir, file))
            file_paths[1].append(args.output_dir)
    print(f"Found {len(file_paths[0])} files to process")
    
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
                        default='/home/lidong1/qilongma/Feynman_LLM/concepts/arc/concepts_OpenAI-GPT-4o-mini', 
                        help='Directory containing ARC dataset concepts')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/lidong1/qilongma/Feynman_LLM/concepts/arc/concepts_OpenAI-GPT-4o-mini_taxo_path_gen', 
                        help='Output directory for generated taxonomy paths')
    parser.add_argument("--concept_model_name", "-cn", type=str, 
                        default="OpenAI-GPT-4o-mini", 
                        help='Concept model name to use for generating taxonomy paths')
    parser.add_argument('--max_workers', type=int, 
                        default=8, 
                        help='Number of workers to use for processing files in parallel')
    args = parser.parse_args()
    
    # Run the process
    process_arc_files(args)
