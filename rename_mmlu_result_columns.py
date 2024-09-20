import os
import pandas as pd

# 指定文件夹路径
folder_path = "/home/lidong1/qilongma/Feynman_LLM/results/mmlu/results_Meta-Llama-3-8B_feynman_OpenAI-GPT-4o-mini_question_concept"

# 目标列名
new_columns = [
    'questions', 'A', 'B', 'C', 'D', 'answers', 
    'Meta-Llama-3-8B_correct', 'Meta-Llama-3-8B_choiceA_probs', 
    'Meta-Llama-3-8B_choiceB_probs', 'Meta-Llama-3-8B_choiceC_probs', 
    'Meta-Llama-3-8B_choiceD_probs'
]

# 遍历文件夹中的所有CSV文件
cnt = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # 使用pandas打开CSV文件
        df = pd.read_csv(file_path)
        
        # 更改列名
        if len(df.columns) == len(new_columns):
            df.columns = new_columns
            
            # 保存更改后的CSV文件
            df.to_csv(file_path, index=False)
            print(f"Updated columns in: {filename}")
            cnt += 1
        else:
            print(f"Skipping {filename} due to column count mismatch.")

print(f"All {cnt} files have been processed.")
