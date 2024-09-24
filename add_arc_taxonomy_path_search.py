import time
import os
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


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

def have_common_words(phrase1, phrase2):
    # 将短语转换为小写并拆分成单词列表
    words1 = set(phrase1.lower().replace("_", " ").split())
    words2 = set(phrase2.lower().replace("_", " ").split())
    
    # 检查两个集合是否有共同的单词
    common_words = words1.intersection(words2)
    
    # 返回是否存在共同单词
    return len(common_words) > 0

# Global variable to store taxonomy
taxonomy = {}

def load_taxonomy(taxonomy_dir):
    global taxonomy
    for taxonomy_file in sorted(os.listdir(taxonomy_dir)):
        if taxonomy_file.endswith('.jsonl'):
            discipline = taxonomy_file.split('.')[0]
            with open(os.path.join(taxonomy_dir, taxonomy_file), 'r') as f:
                taxonomy[discipline] = [json.loads(line) for line in f]

def related(discip_name, subj_name, cls_name, kp_name, q_with_a, concept):
    concept_match_kp = concept.lower() in kp_name.lower() or (kp_name.lower() in concept.lower() and len(kp_name) > 3)
    # subj_match_taxonomy = (discip_name.lower() in q_with_a.lower() or q_with_a.lower() in discip_name.lower()) or \
    #                     (subj_name.lower() in q_with_a.lower() or q_with_a.lower() in subj_name.lower()) or \
    #                     (cls_name.lower() in q_with_a.lower() or q_with_a.lower() in cls_name.lower()) or \
    #                     (kp_name.lower() in q_with_a.lower() or q_with_a.lower() in kp_name.lower())
    ques_match_taxonomy = have_common_words(q_with_a, discip_name) or \
                        have_common_words(q_with_a, subj_name) or \
                        have_common_words(q_with_a, cls_name) or \
                        have_common_words(q_with_a, kp_name)
    return concept_match_kp and ques_match_taxonomy

def find_taxonomy_path(question, choices, answer, level, grade, concept):
    global taxonomy
    q_with_a = format_example(question, choices, answer)
    for discipline, subjects in taxonomy.items():
        for subject in subjects:
            for class_session in subject['topic_knowledge_points']:
                for knowledge_point in class_session['knowledge_points']:
                    if related(discipline, subject['subject_name'], class_session['class_session'], knowledge_point, q_with_a, concept):
                        return f"{concept}: {discipline} -> {subject['subject_name']} -> {class_session['class_session']} -> {knowledge_point}"
                    # if concept.lower() in knowledge_point.lower() and mmlu_subject.lower() in subject['subject_name'].lower():
                    #     return f"{concept}: {discipline} -> {subject['subject_name']} -> {class_session['class_session']} -> {knowledge_point}"
    return f"{concept}: Not found"

def process_arc_file(file_path, output_split_dir):
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
                    path = find_taxonomy_path(question, choices, answer, level, grade, concept)
                    paths.append(path if path is not None else concept + ": Not found")
                    cnt_tot_file += 1
                    if 'Not found' in path:
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

def process_arc_files(concept_dir, taxonomy_dir, output_dir, max_workers=16):
    # Load all taxonomy files
    tic = time.time()
    load_taxonomy(taxonomy_dir)
    toc = time.time()
    print(f"Loaded taxonomy in {toc - tic:.3f} seconds")
    
    # Collect all file paths
    file_paths = [[], []]
    for file in sorted(os.listdir(concept_dir)):
        if file.startswith('ARC-') and file.endswith('_concepts.jsonl'):
            file_paths[0].append(os.path.join(concept_dir, file))
            file_paths[1].append(output_dir)
    print(f"Found {len(file_paths[0])} files to process")
    
    # Use multiprocessing to process files in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_arc_file, file_paths[0], file_paths[1]))
    
    cnt_404, cnt_tot = map(sum, zip(*results))
    print(f"\nFinal - 404: {cnt_404}, total: {cnt_tot}, 404 rate: {cnt_tot and cnt_404 / cnt_tot or 'N/A'}")

# Parameters
concept_dir = '/home/lidong1/qilongma/Feynman_LLM/concepts/arc/concepts_OpenAI-GPT-4o-mini'       # Replace with your MMLU dataset concept directory
taxonomy_dir = '/home/lidong1/qilongma/taxonomy/cleaned_subj_cls_kps'  # Replace with your taxonomy directory
output_dir = '/home/lidong1/qilongma/Feynman_LLM/concepts/arc/concepts_OpenAI-GPT-4o-mini_taxo_path_search'      # Replace with your desired output directory

# Run the process
process_arc_files(concept_dir, taxonomy_dir, output_dir, max_workers=32)
