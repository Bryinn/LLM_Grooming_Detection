import ast
import os
MODEL_IDS_PATH = os.path.join(os.path.dirname(__file__), 'model_ids.py')
with open(MODEL_IDS_PATH, 'r', encoding='utf-8') as f:
    MODEL_ID_LIST = ast.literal_eval(f.read())

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
    for conv_id, conv_msgs in test_convs.items():
        try:
            conv_text = "\n".join(conv_msgs)
            prompt = (initial_prompt or "") + f"\nCONVERSATION:\n{conv_text}\n\nNow, based only on the above conversation, respond strictly with a single valid JSON object in this format: {{\"conversation_id\": {conv_id}, \"is_predatory\": true/false, \"reasoning\": \"...\"}}. Do not include any other text."
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
            print(f"[DEBUG] conv_id: {conv_id}, input_ids.shape: {input_ids.shape}")
            print(f"[DEBUG] input_ids: {input_ids}")
            if input_ids.numel() == 0:
                print(f"[DEBUG] Skipping conversation {conv_id}: input_ids is empty!")
                continue
            output = model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            if debug_count < debug_print_limit:
                print(f"[DEBUG] Raw model output for conv_id {conv_id}:\n{decoded}\n{'-'*60}")
                debug_count += 1
            json_match = re.search(r'\{.*\}', decoded, re.DOTALL)
            if json_match:
                try:
                    pred = json.loads(json_match.group(0).replace("'", '"'))
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
