from pathlib import Path
import pandas as pd
import json

def compare_experiment_results(exp_name1, exp_name2, file1, file2, output_csv):
    # Load JSON files
    with open(file1, 'r') as f:
        results1 = json.load(f)
    with open(file2, 'r') as f:
        results2 = json.load(f)

    # Initialize the DataFrame
    data = []

    # Compare categories
    for category in sorted(results1['categories'].keys()):
        result1 = results1['categories'][category]
        result2 = results2['categories'][category]
        diff = result2 - result1
        is_promoted = diff > 0
        data.append([category, "", "", result1, result2, diff, is_promoted])

    # Compare subcategories
    for subcategory in sorted(results1['subcategories'].keys()):
        result1 = results1['subcategories'][subcategory]
        result2 = results2['subcategories'][subcategory]
        diff = result2 - result1
        is_promoted = diff > 0
        data.append(["", subcategory, "", result1, result2, diff, is_promoted])

    # Compare subjects
    for subject in sorted(results1['subjects'].keys()):
        result1 = results1['subjects'][subject]
        result2 = results2['subjects'][subject]
        diff = result2 - result1
        is_promoted = diff > 0
        data.append(["", "", subject, result1, result2, diff, is_promoted])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["category", "subcategory", "subject",
                                     f"{exp_name1}_result", f"{exp_name2}_result", 
                                     "diff", "is_promoted"])

    # Save to CSV
    df.to_csv(output_csv, index=False)

# Example usage
results_dir = Path("/home/lidong1/qilongma/mmlu-master/results")
exp_name1 = "Meta-Llama-2-7B"
exp_name2 = "Meta-Llama-2-7B_feynman_OpenAI-GPT-4o-mini_sep_question_concept_taxo_path_gen_4th"
output_csv_name = "comparison_llama2-7b&gpt4o-mini_sep_taxo_gen_4th.csv"
compare_experiment_results(
    exp_name1,
    exp_name2,
    results_dir / f"accuracies_{exp_name1}.json",
    results_dir / f"accuracies_{exp_name2}.json",
    results_dir / output_csv_name
)
