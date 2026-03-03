
# Utility and summary functions for LLM Grooming Detection
import ast
import pandas as pd

def write_accuracy_summary(results_file, summary_file):
    """Aggregate results and write accuracy summary comparing models."""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(ast.literal_eval(line.strip()))
            except Exception:
                continue
    total = len(results)
    correct = sum(1 for r in results if 'label' in r and r['is_predatory'] == r['label'])
    accuracy = correct / total if total else 0.0
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Total: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
    print(f"Summary written to {summary_file}")

def prepare_text_data(load_pj_dataset, load_pan12_training, PJ_DIR, PAN12_TRAIN_PATH):
    pj_df = load_pj_dataset(PJ_DIR)
    try:
        pan12_df = load_pan12_training(PAN12_TRAIN_PATH)
    except Exception as e:
        print(f"Could not load pan12-training: {e}")
        pan12_df = None
    dfs = [pj_df]
    if pan12_df is not None:
        dfs.append(pan12_df)
    all_df = pd.concat(dfs, ignore_index=True)
    texts = all_df['text'].dropna().astype(str).tolist()
    return texts
