import os

def get_model_and_tokenizer(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"[DEBUG] Loaded tokenizer from {model_path}")
    print(f"[DEBUG] Tokenizer vocab size: {getattr(tokenizer, 'vocab_size', 'N/A')}")
    print(f"[DEBUG] Tokenizer special tokens: bos={getattr(tokenizer, 'bos_token', None)}, eos={getattr(tokenizer, 'eos_token', None)}, pad={getattr(tokenizer, 'pad_token', None)}")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    return model, tokenizer, device

def evaluate_causal_lm(model_path, test_convs, initial_prompt=None, results_file=None, temperature=1.0, top_p=0.9, model_label=None):
    import re, json
    model, tokenizer, device = get_model_and_tokenizer(model_path)
    debug_print_limit = 5
    debug_count = 0
    delimiter = "\n### RESPONSE:\n"
    for conv_id, conv_msgs in test_convs.items():
        try:
            conv_text = "\n".join(conv_msgs)
            prompt = (initial_prompt or "") + f"\nCONVERSATION:\n{conv_text}\n\nNow, based only on the above conversation, respond strictly with a single valid JSON object in this format: {{\"conversation_id\": {conv_id}, \"is_predatory\": true/false, \"reasoning\": \"...\"}}. Do not include any other text." + delimiter
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)
            print(f"[DEBUG] conv_id: {conv_id}, input_ids.shape: {input_ids.shape}")
            print(f"[DEBUG] input_ids: {input_ids}")
            if input_ids.numel() == 0:
                print(f"[DEBUG] Skipping conversation {conv_id}: input_ids is empty!")
                continue
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                temperature=temperature,
                top_p=top_p,
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
                print(f"[DEBUG] Raw model output for conv_id {conv_id}:\n{decoded}\n{'-'*60}")
                debug_count += 1
            # JSON extraction: replace single quotes, strip whitespace, and use regex
            import re
            json_match = re.search(r'\{[\s\S]*?\}', decoded)
            json_str = None
            if json_match:
                json_str = json_match.group(0).replace("'", '"').strip()
            if json_str:
                try:
                    pred = json.loads(json_str)
                except Exception:
                    pred = {"conversation_id": int(conv_id), "is_predatory": None, "reasoning": "Could not parse model output as JSON."}
            else:
                pred = {"conversation_id": int(conv_id), "is_predatory": None, "reasoning": "No JSON found in model output."}
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


def evaluate_model(model_path, test_convs, initial_prompt=None, results_file=None, temperature=1.0, top_p=0.9):
    # Dynamically evaluate any model by its model_id
    model_id = os.path.basename(model_path).replace('_finetuned', '')
    evaluate_causal_lm(model_path, test_convs, initial_prompt, results_file, temperature, top_p, model_label=model_id)

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
def run_evaluation_summarizer(results_dir):
    import os
    import glob
    import pandas as pd
    # Find all result files in all model subfolders
    all_results = []
    for model_folder in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_folder)
        if not os.path.isdir(model_path):
            continue
        for file in glob.glob(os.path.join(model_path, '*.txt')):
            if 'summary' in file:
                continue
            all_results.append((model_folder, file))
    # Load ground truth
    # (Assume PAN12_TEST_PATH is available as global or hardcode path)
    try:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from data_loader import load_pan12_test_ground_truth
        PAN12_TEST_PATH = os.path.join('filtered_datasets', 'pan12-test', 'pan12-sexual-predator-identification-test-corpus-2012-05-17.json')
        gt_df = load_pan12_test_ground_truth(PAN12_TEST_PATH)
        gt = gt_df.set_index('conversation_id').to_dict()['is_predatory']
    except Exception as e:
        print(f"Could not load ground truth: {e}")
        gt = None
    # Compute stats for each result file
    stats = []
    for model_name, file in all_results:
        preds = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = eval(line.strip()) if line.strip().startswith('{') else None
                except Exception:
                    d = None
                if d and 'conversation_id' in d and 'is_predatory' in d:
                    preds.append(d)
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
            stats.append({
                'model': model_name,
                'file': os.path.basename(file),
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'accuracy': acc,
                'settings': os.path.basename(file).replace(model_name, '').replace('.txt', '')
            })
    # Compare models/settings
    if stats:
        df = pd.DataFrame(stats)
        df.to_csv(os.path.join(results_dir, 'all_model_stats.csv'), index=False)
        print("Model evaluation statistics written to all_model_stats.csv")
        # Print best by accuracy
        print(df.sort_values('accuracy', ascending=False).head(10))
