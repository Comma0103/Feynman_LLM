import time
import os
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor

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

def related(discip_name, subj_name, cls_name, kp_name, mmlu_subj, concept):
    concept_match_kp = concept.lower() in kp_name.lower() or (kp_name.lower() in concept.lower() and len(kp_name) > 3)
    # subj_match_taxonomy = (discip_name.lower() in mmlu_subj.lower() or mmlu_subj.lower() in discip_name.lower()) or \
    #                     (subj_name.lower() in mmlu_subj.lower() or mmlu_subj.lower() in subj_name.lower()) or \
    #                     (cls_name.lower() in mmlu_subj.lower() or mmlu_subj.lower() in cls_name.lower()) or \
    #                     (kp_name.lower() in mmlu_subj.lower() or mmlu_subj.lower() in kp_name.lower())
    subj_match_taxonomy = have_common_words(mmlu_subj, discip_name) or \
                        have_common_words(mmlu_subj, subj_name) or \
                        have_common_words(mmlu_subj, cls_name) or \
                        have_common_words(mmlu_subj, kp_name)
    return concept_match_kp and subj_match_taxonomy

def find_taxonomy_path(mmlu_subject, concept):
    global taxonomy
    for discipline, subjects in taxonomy.items():
        for subject in subjects:
            for class_session in subject['topic_knowledge_points']:
                for knowledge_point in class_session['knowledge_points']:
                    if related(discipline, subject['subject_name'], class_session['class_session'], knowledge_point, mmlu_subject, concept):
                        return f"{concept}: {discipline} -> {subject['subject_name']} -> {class_session['class_session']} -> {knowledge_point}"
                    # if concept.lower() in knowledge_point.lower() and mmlu_subject.lower() in subject['subject_name'].lower():
                    #     return f"{concept}: {discipline} -> {subject['subject_name']} -> {class_session['class_session']} -> {knowledge_point}"
    return f"{concept}: Not found"

def process_mmlu_file(file_path, output_split_dir):
    try:
        cnt_404_file, cnt_tot_file = 0, 0
        
        tic = time.time()
        
        df = pd.read_csv(file_path)
        mmlu_subject = os.path.basename(file_path).split('_concepts.csv')[0]

        taxonomy_paths = []
        for i in range(len(df)):
            if pd.isna(df.loc[i, "concepts"]):
                taxonomy_paths.append(None)
            else:
                concepts = df.loc[i, "concepts"].split(", ")
                paths = []
                for concept in concepts:
                    path = find_taxonomy_path(mmlu_subject, concept)
                    paths.append(path)
                    cnt_tot_file += 1
                    if "Not found" in path:
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

def process_mmlu_files(concept_dir, taxonomy_dir, output_dir, max_workers=16):
    # Load all taxonomy files
    tic = time.time()
    load_taxonomy(taxonomy_dir)
    toc = time.time()
    print(f"Loaded taxonomy in {toc - tic:.3f} seconds")
    
    # Collect all file paths
    file_paths = [[], []]
    for split in ['dev', 'val', 'test']:
        split_dir = os.path.join(concept_dir, split)
        for file in sorted(os.listdir(split_dir)):
            if file.endswith('_concepts.csv'):
                file_paths[0].append(os.path.join(split_dir, file))
                file_paths[1].append(os.path.join(output_dir, split))
    print(f"Found {len(file_paths[0])} files to process")
    
    # Use multiprocessing to process files in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_mmlu_file, file_paths[0], file_paths[1]))
    
    cnt_404, cnt_tot = map(sum, zip(*results))
    print(f"\nFinal - 404: {cnt_404}, total: {cnt_tot}, 404 rate: {cnt_tot and cnt_404 / cnt_tot or 'N/A'}")

# Parameters
concept_dir = '/home/lidong1/qilongma/mmlu-master/concepts/concepts_OpenAI-GPT-4o-mini'       # Replace with your MMLU dataset concept directory
taxonomy_dir = '/home/lidong1/qilongma/taxonomy/cleaned_subj_cls_kps'  # Replace with your taxonomy directory
output_dir = '/home/lidong1/qilongma/mmlu-master/concepts/concepts_OpenAI-GPT-4o-mini_taxo_path_search'      # Replace with your desired output directory

# Run the process
process_mmlu_files(concept_dir, taxonomy_dir, output_dir, max_workers=32)
