import os
import pandas as pd
import re

def process_concepts(concepts_str):
    if not isinstance(concepts_str, str):
        return concepts_str
    
    # 提取冒号前的部分（即概念名称）
    concepts_list = [re.split(r':', concept, maxsplit=1)[0].strip() for concept in concepts_str.split('\n')]

    # 去掉包含 "these concepts" 或不以数字开头的行
    concepts_list = [concept for concept in concepts_list if not re.search(r"these concepts", concept, re.IGNORECASE)]
    concepts_list = [concept for concept in concepts_list if re.match(r"^\d+", concept)]
    
    # 去掉编号和加粗字体的符号
    # concepts_str = re.sub(r"\d+\.\s*\*\*(.*?)\*\*", r"\1", concepts_str)
    concepts_list = [re.sub(r"^\d+\.\s*", "", concept) for concept in concepts_list]
    concepts_list = [re.sub(r"\*\*(.*?)\*\*", r"\1", concept) for concept in concepts_list]

    # 连接成字符串返回
    return ", ".join(concepts_list)

def process_csv_files(input_folder, output_folder):
    # 遍历所有子文件夹
    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                # 构造输入和输出文件路径
                input_file_path = os.path.join(subdir, file)
                output_file_path = os.path.join(output_folder, os.path.relpath(input_file_path, input_folder))
                
                # 创建输出文件夹（如果不存在）
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                
                # 读取csv文件
                df = pd.read_csv(input_file_path)
                
                # 处理concepts列
                df['concepts'] = df['concepts'].apply(process_concepts)
                
                # 保存到新的csv文件
                df.to_csv(output_file_path, index=False)

# 设置输入文件夹和输出文件夹路径
input_folder = '/home/lidong1/qilongma/Feynman_LLM/concepts/mmlu/old/concepts_OpenAI-GPT-4o-mini'
output_folder = '/home/lidong1/qilongma/Feynman_LLM/concepts/mmlu/concepts_OpenAI-GPT-4o-mini_from_old'

# 处理csv文件
process_csv_files(input_folder, output_folder)
