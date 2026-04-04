import json
import pandas as pd
from Globals import PAN12_TRAIN_PATH, PAN12_TEST_PATH, PJ_DIR
from data_loader import load_pan12_with_labels, load_pan12_training, load_pj_dataset, load_pan12_test_ground_truth


def load_eval_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def sample_balanced_conversations(df, pred_col, n_pred, n_non_pred, seed=42):
    pred_df = df[df[pred_col] == True]
    non_pred_df = df[df[pred_col] == False]
    pred_sample = pred_df.sample(n=min(n_pred, len(pred_df)), random_state=seed) if n_pred else pred_df
    non_pred_sample = non_pred_df.sample(n=min(n_non_pred, len(non_pred_df)), random_state=seed) if n_non_pred else non_pred_df
    return pd.concat([pred_sample, non_pred_sample]).sample(frac=1, random_state=seed)

def get_eval_conversations_from_config(config):
    dataset = config.get('dataset', 'pan12-test')
    max_samples = config.get('max_samples')
    balance = config.get('balance')
    seed = config.get('random_seed', 42)
    if dataset == 'pan12-test':
        df = load_pan12_with_labels(PAN12_TEST_PATH)
        pred_col = 'is_predatory'
    elif dataset == 'pan12-training':
        df = load_pan12_with_labels(PAN12_TRAIN_PATH)
        pred_col = 'is_predatory'
    elif dataset == 'PJ':
        df = load_pj_dataset(PJ_DIR)
        pred_col = 'is_predatory'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if balance:
        n_pred = balance.get('predatory', 0)
        n_non_pred = balance.get('non_predatory', 0)
        df = sample_balanced_conversations(df, pred_col, n_pred, n_non_pred, seed)
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed)
    # Convert to dict for evaluation
    if 'messages' in df.columns:
        convs = {row['conversation_id']: [m['text'] for m in row['messages']] for _, row in df.iterrows()}
    elif 'text' in df.columns:
        convs = df.groupby('conversation_id')['text'].apply(list).to_dict()
    else:
        raise ValueError("DataFrame does not have expected columns.")
    return convs
import os
from tqdm import tqdm

def get_model_and_tokenizer(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #print(f"[DEBUG] Loaded tokenizer from {model_path}")
    #print(f"[DEBUG] Tokenizer vocab size: {getattr(tokenizer, 'vocab_size', 'N/A')}")
    #print(f"[DEBUG] Tokenizer special tokens: bos={getattr(tokenizer, 'bos_token', None)}, eos={getattr(tokenizer, 'eos_token', None)}, pad={getattr(tokenizer, 'pad_token', None)}")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    return model, tokenizer, device

def evaluate_causal_lm(model_path, test_convs, results_file=None, temperature=1.0, top_p=1.0, model_label=None):
    import re, json
    model, tokenizer, device = get_model_and_tokenizer(model_path)
    debug_print_limit = 5
    debug_count = 0
    delimiter = "\n### RESPONSE:\n"
    for conv_id, conv_msgs in tqdm(test_convs.items()):
        try:
            conv_text = "\n".join(conv_msgs)
            prompt = f"\nCONVERSATION:\n{conv_text}\n\nNow, based only on the above conversation, respond strictly with a single valid JSON object in this format: {{\"is_predatory\": true/false, \"reasoning\": \"...\"}} as a raw string. Do not include any other text." + delimiter
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)
            #print(f"[DEBUG] conv_id: {conv_id}, input_ids.shape: {input_ids.shape}")
            #print(f"[DEBUG] input_ids: {input_ids}")
            # Ensure temperature and top_p are valid
            safe_temperature = temperature if temperature > 0 else 1e-7
            safe_top_p = top_p if top_p > 0 else 1e-7
            if input_ids.numel() == 0:
                #print(f"[DEBUG] Skipping conversation {conv_id}: input_ids is empty!")
                continue
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                temperature=safe_temperature,
                top_p=safe_top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            )
            full_decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            # Only keep the part after the delimiter
            if delimiter in full_decoded:
                decoded = full_decoded.split(delimiter, 1)[-1].strip()
            else:
                decoded = full_decoded
            eos_token = getattr(tokenizer, 'eos_token', None)
            if eos_token and eos_token in decoded:
                decoded = decoded.split(eos_token)[0]
            if debug_count < debug_print_limit:
                #print(f"[DEBUG] Raw model output for conv_id {conv_id}:\n{decoded}\n{'-'*60}")
                debug_count += 1
            # JSON extraction: replace single quotes, strip whitespace, and use regex
            import re
            # Improved JSON extraction and cleaning
            json_str = None
            pred = None
            # Try to extract the largest valid JSON substring (greedy)
            json_matches = list(re.finditer(r'\{[\s\S]*\}', decoded))
            if json_matches:
                # Prefer the largest match (most likely the full object)
                json_str = max((m.group(0) for m in json_matches), key=len).strip()
            # Try normal JSON parse
            if json_str:
                try:
                    pred = json.loads(json_str)
                except Exception:
                    # Try to fix common issues: single quotes, trailing commas, newlines, extra text
                    try:
                        fixed = json_str.replace("'", '"')
                        fixed = re.sub(r',\s*}', '}', fixed)
                        fixed = re.sub(r',\s*]', ']', fixed)
                        fixed = re.sub(r'\n', ' ', fixed)
                        # Remove text before first { and after last }
                        fixed = re.sub(r'^.*?(\{)', r'\1', fixed)
                        fixed = re.sub(r'(\}).*$', r'\1', fixed)
                        pred = json.loads(fixed)
                    except Exception:
                        # Try even more aggressive cleaning: remove all non-JSON chars before/after
                        try:
                            fixed2 = re.sub(r'^[^\{]*', '', decoded)
                            fixed2 = re.sub(r'[^\}]*$', '', fixed2)
                            fixed2 = fixed2.replace("'", '"')
                            fixed2 = re.sub(r',\s*}', '}', fixed2)
                            fixed2 = re.sub(r',\s*]', ']', fixed2)
                            fixed2 = re.sub(r'\n', ' ', fixed2)
                            pred = json.loads(fixed2)
                        except Exception:
                            pred = None
            if not pred:
                # Log failure for review with more context
                fail_log = results_file + '.failures' if results_file else 'eval_failures.txt'
                with open(fail_log, 'a', encoding='utf-8') as f:
                    f.write(f'conv_id={conv_id} | prompt="{prompt[:200]}..." | raw="{decoded}"")\n')
                pred = {"conversation_id": int(conv_id), "is_predatory": None, "reasoning": "Could not parse model output as JSON."}
            else:
                pred["conversation_id"] = int(conv_id)
            if results_file:
                with open(results_file, 'a', encoding='utf-8') as f:
                    f.write(str(pred) + '\n')
            else:
                print(pred)
        except Exception as e:
            label = model_label or model_path
            import traceback
            print(f"Error evaluating {label} for conversation {conv_id}: {e}")
            traceback.print_exc()
# Model evaluation functions for LLM Grooming Detection
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_model(model_path, test_convs, results_file=None, temperature=1.0, top_p=0.9):
    # Dynamically evaluate any model by its model_id
    model_id = os.path.basename(model_path).replace('_finetuned', '')
    evaluate_causal_lm(model_path, test_convs, results_file, temperature, top_p, model_label=model_id)

def evaluate_companionv1(test_convs, initial_prompt=None, results_file=None):
    for conv_id, conv_msgs in test_convs.items():
        try:
            conv_text = "\n".join(conv_msgs)
            prompt = (initial_prompt or "") + f"\nCONVERSATION:\n{conv_text}\nRespond in JSON: {{conversation_id, is_predatory, reasoning}}"
            pred = {
                "conversation_id": int(conv_id),
                "is_predatory": bool(np.random.randint(0, 2)),
                "reasoning": "Random stub reasoning."
            }
            if results_file:
                with open(results_file, 'a', encoding='utf-8') as f:
                    f.write(str(pred) + '\n')
            else:
                print(pred)
        except Exception as e:
            print(f"Error evaluating compAnIonv1 for conversation {conv_id}: {e}")

# Summarizer process for evaluation results
def run_evaluation_summarizer(results_dir, settings_filters=None, model_name_filters=None):
    # No user dialogue here; all filter input should be handled by caller (e.g. LLM.py)
    import os
    import glob
    import pandas as pd
    # Find all result files in all model subfolders
    all_results = []
    for model_folder in os.listdir(results_dir):
        # Model name filter
        if model_name_filters:
            if not all(f in model_folder for f in model_name_filters):
                continue
        model_path = os.path.join(results_dir, model_folder)
        if not os.path.isdir(model_path):
            continue
        for file in glob.glob(os.path.join(model_path, '*.txt')):
            if 'summary' in file:
                continue
            settings_str = os.path.basename(file).replace(model_folder, '').replace('.txt', '')
            if settings_filters:
                if not all(f in settings_str for f in settings_filters):
                    continue
            all_results.append((model_folder, file))
    # Load ground truth
    try:
        import sys
        sys.path.append(os.path.dirname(__file__))
        gt_df = load_pan12_test_ground_truth(PAN12_TEST_PATH)
        gt = gt_df.set_index('conversation_id').to_dict()['is_predatory']
    except Exception as e:
        print(f"Could not load ground truth: {e}")
        gt = None
    # Compute stats for each result file
    stats = []
    for model_name, file in all_results:
        preds = []
        num_erroneous = 0
        total_preds = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                total_preds += 1
                try:
                    d = eval(line.strip()) if line.strip().startswith('{') else None
                except Exception:
                    d = None
                if d and 'conversation_id' in d and 'is_predatory' in d:
                    preds.append(d)
                    if d['is_predatory'] is None:
                        num_erroneous += 1
                else:
                    num_erroneous += 1
        # Compute metrics if ground truth available
        if gt:
            tp = fp = tn = fn = 0
            for d in preds:
                cid = d['conversation_id']
                pred = d['is_predatory']
                true = gt.get(cid, None)
                if pred is None or true is None:
                    continue
                if pred and true:
                    tp += 1
                elif pred and not true:
                    fp += 1
                elif not pred and not true:
                    tn += 1
                elif not pred and true:
                    fn += 1
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            fail_rate = num_erroneous / total_preds if total_preds > 0 else 0
            stats.append({
                'model': model_name,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_erroneous': num_erroneous,
                'fail_rate': fail_rate,
                'settings': os.path.basename(file).replace(model_name, '').replace('.txt', '')
            })
    # Compare models/settings
    if stats:
        df = pd.DataFrame(stats)
        # Compose output file name with filters
        name_parts = ["model_stats"]
        if settings_filters:
            name_parts.extend(settings_filters)
        if model_name_filters:
            name_parts.extend(["model_" + f for f in model_name_filters])
        name = "_".join(name_parts) + ".csv"
        df.to_csv(os.path.join(results_dir, name), index=False)
        print(f"Model evaluation statistics written to {name}")
        # Print best by F1 score
        print("Top models by F1 score:")
        print(df.sort_values('f1', ascending=False))
        print("\nFailure rates:")
        print(df[['model', 'fail_rate', 'num_erroneous', 'settings']].sort_values('fail_rate', ascending=False).head(10))