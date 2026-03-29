

import ast
import os
from data_loader import load_pj_dataset, load_test_convs, load_pan12_test, load_pan12_training
from evaluation import evaluate_companionv1, run_evaluation_summarizer
from training import train_hf_llm
from utils import select_models, check_existing_result_folders, get_already_evaluated_conversation_ids

from Globals import MODEL_IDS, PAN12_TRAIN_PATH, PAN12_TEST_PATH, PJ_DIR, MAX_EVAL_THREADS
GROUPED_TRAIN_PATH = os.path.join('filtered_datasets', 'pan12-training', 'grouped_cases.json')
GROUPED_TEST_PATH = os.path.join('filtered_datasets', 'pan12-test', 'grouped_cases.json')
MODELS_DIR = 'models'

model_ids = MODEL_IDS

def eval_worker(model_name, models_dir, results_dir, test_convs, temperature, top_p):
    import os
    from evaluation import evaluate_model
    model_path = os.path.join(models_dir, model_name)
    model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    settings_str = f"temp{temperature}_top{top_p}"
    results_file = os.path.join(model_results_dir, f"{model_name}_{settings_str}{'_short' if len(test_convs) < 100 else ''}.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('')
    evaluate_model(model_path, test_convs, results_file=results_file, temperature=temperature, top_p=top_p)
    print(f"Evaluation metrics and outputs written to {results_file}")

MODEL_IDS_PATH = os.path.join(os.path.dirname(__file__), 'model_ids.py')
with open(MODEL_IDS_PATH, 'r', encoding='utf-8') as f:
    MODEL_ID_LIST = ast.literal_eval(f.read())

# Unified LLM Training Script
# compAnIonv1

# Paths

os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    # User interaction: train or evaluate
    print("Select mode:")
    print("1. Train models")
    print("2. Evaluate models")
    print("3. Delete models")
    print("4. Add untrained model to models directory")
    mode_choice = input("Enter 1, 2, 3, or 4: ").strip()

    dev_limit = 1000
    # Few-shot examples for more robust output
    initial_prompt = (
        "[INST] You are a helpful assistant trained to provide sound judgement on predatory conversation in the context of child predators targeting children in sexual ways.\n"
        "You must answer in JSON format: {is_predatory, reasoning}.\n"
        "Given the following conversation, respond strictly in this JSON format.\n"
        "[/INST]"
    )
    if mode_choice == "4":
        print("\n--- Add non-finetuned models from model_ids ---")
        print("Available base models:")
        base_names = [mid.split('/')[-1] for mid in MODEL_ID_LIST if not mid.strip().startswith('#')]
        for idx, name in enumerate(base_names, 1):
            print(f"{idx}. {name}")
        print(f"{len(base_names)+1}. All models")
        model_choice = input(f"Which model(s) do you want to add? (Enter number or comma-separated list, or {len(base_names)+1} for all): ").strip()
        if model_choice == str(len(base_names)+1):
            to_add = base_names
        else:
            try:
                indices = [int(i.strip())-1 for i in model_choice.split(",")]
                to_add = [base_names[i] for i in indices if 0 <= i < len(base_names)]
            except:
                print("Invalid selection. Exiting.")
                return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        for model_name in to_add:
            model_dir = os.path.join(MODELS_DIR, f"{model_name}_non_finetuned")
            if os.path.exists(model_dir):
                print(f"Model directory already exists: {model_dir}")
            else:
                try:
                    print(f"Downloading model and tokenizer for {model_name}...")
                    # Find full model_id from MODEL_ID_LIST
                    model_id_full = next((mid for mid in MODEL_ID_LIST if mid.split('/')[-1] == model_name), None)
                    if not model_id_full:
                        print(f"Could not find model_id for {model_name} in MODEL_ID_LIST.")
                        continue
                    tokenizer = AutoTokenizer.from_pretrained(model_id_full)
                    model = AutoModelForCausalLM.from_pretrained(model_id_full)
                    os.makedirs(model_dir)
                    model.save_pretrained(model_dir, safe_serialization=True)
                    tokenizer.save_pretrained(model_dir)
                    print(f"Created and populated directory for non-finetuned model: {model_dir}")
                except Exception as e:
                    print(f"[ERR] Failed to download or save model/tokenizer for {model_name}: {e}")
        print("Non-finetuned models added to models directory.")
        return
    elif mode_choice == "1":
        print("Select training mode:")
        print("1. Standard model selection")
        print("2. Load training parameters from statefile")
        train_mode = input("Enter 1 or 2: ").strip()
        print("Select dataset size for training:")
        print("1. Development (first 1000 examples)")
        print("2. Full dataset")
        size_choice = input("Enter 1 or 2: ").strip()
        use_dev = size_choice == "1"
        if train_mode == "2":
            import json
            statefile_path = input("Enter path to statefile (default: train_state.json): ").strip() or "train_state.json"
            try:
                with open(statefile_path, 'r', encoding='utf-8') as sf:
                    state_entries = json.load(sf)
                if not isinstance(state_entries, list):
                    state_entries = [state_entries]
                grouped_df = load_pan12_training(GROUPED_TRAIN_PATH, max_records=dev_limit if use_dev else 0)
                cases = grouped_df.to_dict(orient='records')
                print(f"[DEBUG] Number of cases loaded for training: {len(cases)}")
                custom_name = input("Add a custom name to the model output folder? (leave blank for default): ").strip()
                suffix = f"_{custom_name}" if custom_name else ""
                mode_str = "dev" if use_dev else "full"
                for entry in state_entries:
                    model_id = entry.get("model_id")
                    epoch = entry.get("epochs", 1)
                    learning_rate = entry.get("learning_rate", 2e-5)
                    output_dir = os.path.join(
                        MODELS_DIR,
                        f"{model_id.split('/')[-1]}_{mode_str}_ep{epoch}_lr{learning_rate}{suffix}_finetuned"
                    )
                    if os.path.exists(output_dir):
                        print(f"[SKIP] Model already exists: {output_dir}")
                        continue
                    print(f"[TRAIN] model_id={model_id}, epochs={epoch}, learning_rate={learning_rate}")
                    train_hf_llm(
                        cases,
                        model_id=model_id,
                        initial_prompt=initial_prompt,
                        output_dir=output_dir,
                        suffix=suffix,
                        default_dir_name=model_id.split('/')[-1],
                        epoch=epoch,
                        learning_rate=learning_rate
                    )
            except Exception as e:
                print(f"Failed to load or parse statefile: {e}")
                return
        else:
            grouped_df = load_pan12_training(GROUPED_TRAIN_PATH, max_records=dev_limit if use_dev else 0)
            cases = grouped_df.to_dict(orient='records')
            print(f"[DEBUG] Number of cases loaded for training: {len(cases)}")
            selected_models = select_models(MODEL_ID_LIST, prompt_all='All')
            if not selected_models:
                return
            try:
                epoch = int(input("Enter number of epochs (e.g. 1-10, default 1): ") or "1")
            except Exception:
                print("Invalid input. Using default: 1 epoch.")
                epoch = 1
            lr_choices = [2e-7, 5e-5, 1e-4, 2e-4]
            print("Select learning rate:")
            for i, lr in enumerate(lr_choices, 1):
                print(f"{i}. {lr}")
            lr_choice = input(f"Enter 1-{len(lr_choices)} (default 1): ").strip()
            try:
                learning_rate = lr_choices[int(lr_choice)-1] if lr_choice.isdigit() and 1 <= int(lr_choice) <= len(lr_choices) else lr_choices[0]
            except Exception:
                learning_rate = lr_choices[0]
            custom_name = input("Add a custom name to the model output folder? (leave blank for default): ").strip()
            suffix = f"_{custom_name}" if custom_name else ""
            mode_str = "dev" if use_dev else "full"
            for model_id in selected_models:
                output_dir = os.path.join(
                    MODELS_DIR,
                    f"{model_id.split('/')[-1]}_{mode_str}_ep{epoch}_lr{learning_rate}{suffix}_finetuned"
                )
                train_hf_llm(
                    cases,
                    model_id=model_id,
                    initial_prompt=initial_prompt,
                    output_dir=output_dir,
                    suffix=suffix,
                    default_dir_name=model_id.split('/')[-1],
                    epoch=epoch,
                    learning_rate=learning_rate
                )
    elif mode_choice == "2":
        print("1. Evaluate new models on the pan12-test dataset.")
        print("2. Run evaluation summarizer to get an overview of existing results.")
        print("3. Continue an interrupted evaluation.")
        mode_choice = int(input("Select option: ").strip())
        top_p = 1.0
        temperature = 1.0
        if mode_choice == 2:
            run_evaluation_summarizer("results")
            return
        elif mode_choice == 3:
            test_50 = input("Continue only the first 50 samples? (y/N): ").strip().lower() == 'y'
            try:
                temperature = float(input("Set temperature 0-1 (default 1.0, higher=more creative): ") or "1.0")
                top_p = float(input("Set top_p 0-1 (default 1.0, lower=more focused): ") or "1.0")
            except Exception:
                print("Invalid input. Using default values: temperature=1.0, top_p=1.0")
            all_models = [d for d in os.listdir(MODELS_DIR) if d.endswith("_finetuned") and os.path.isdir(os.path.join(MODELS_DIR, d))]
            if not all_models:
                print("No models found in the models directory.")
                return
            print("\nAvailable models:")
            for idx, m in enumerate(all_models, 1):
                model_id = next((mid for mid in MODEL_ID_LIST if mid.split('/')[-1] in m), "Unknown")
                print(f"{idx}. {m} (model_id: {model_id})")
            print(f"{len(all_models)+1}. All models")
            model_choice = input(f"Which model(s) do you want to continue? (Enter number or comma-separated list, or {len(all_models)+1} for all): ").strip()
            if model_choice == str(len(all_models)+1):
                to_eval = all_models
            else:
                try:
                    indices = [int(i.strip())-1 for i in model_choice.split(",")]
                    to_eval = [all_models[i] for i in indices if 0 <= i < len(all_models)]
                except:
                    print("Invalid selection. Exiting.")
                    return
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            print("\n--- Continuing evaluation on pan12-test dataset ---")
            test_convs = load_test_convs(PAN12_TEST_PATH, test_50)
            if test_convs:
                import subprocess
                import time
                max_parallel = MAX_EVAL_THREADS
                running = []  # List of (model_name, process)
                to_eval_queue = []
                model_remain_map = {}
                for m in to_eval:
                    model_results_dir = os.path.join(results_dir, m)
                    settings_str = f"temp{temperature}_top{top_p}{'_short' if len(test_convs) < 100 else ''}"
                    results_file = os.path.join(model_results_dir, f"{m}_{settings_str}.txt")
                    already_done = get_already_evaluated_conversation_ids(results_file)
                    if already_done:
                        print(f"{m}: Found {len(already_done)} already evaluated conversations. Will continue from there.")
                    else:
                        print(f"{m}: No previous results found. Will start from scratch.")
                    remaining_convs = {cid: msgs for cid, msgs in test_convs.items() if cid not in already_done}
                    if not remaining_convs:
                        print(f"{m}: All conversations already evaluated. Skipping.")
                        continue
                    model_remain_map[m] = remaining_convs
                    to_eval_queue.append(m)
                while to_eval_queue or running:
                    # Start new evaluations if slots available
                    while len(running) < max_parallel and to_eval_queue:
                        m = to_eval_queue.pop(0)
                        args = [
                            'python', __file__, '--eval-worker-continue', m, MODELS_DIR, results_dir,
                            str(temperature), str(top_p), 'append'
                        ]
                        proc = subprocess.Popen(args)
                        running.append((m, proc))
                        print(f"Started continued evaluation for {m} (PID: {proc.pid})")
                    # Check for finished processes
                    for idx in range(len(running)-1, -1, -1):
                        m, proc = running[idx]
                        ret = proc.poll()
                        if ret is not None:
                            print(f"Continued evaluation finished for {m} (PID: {proc.pid}, exit code: {ret})")
                            running.pop(idx)
                    time.sleep(2)
                print("All continued evaluations complete.")
            else:
                print("No test data available for evaluation.")
            return
        elif mode_choice != 1:
            print("Invalid selection.")
            return
        else:
            test_50 = input("Evaluate only the first 50 samples? (y/N): ").strip().lower() == 'y'
            try:
                temperature = float(input("Set temperature 0-1 (default 1.0, higher=more creative): ") or "1.0")
                top_p = float(input("Set top_p 0-1 (default 1.0, lower=more focused): ") or "1.0")
            except Exception:
                print("Invalid input. Using default values: temperature=1.0, top_p=1.0")
            all_models = [d for d in os.listdir(MODELS_DIR) if d.endswith("_finetuned") and os.path.isdir(os.path.join(MODELS_DIR, d))]
            if not all_models:
                print("No models found in the models directory.")
                return
            print("\nAvailable models:")
            for idx, m in enumerate(all_models, 1):
                model_id = next((mid for mid in MODEL_ID_LIST if mid.split('/')[-1] in m), "Unknown")
                print(f"{idx}. {m} (model_id: {model_id})")
            print(f"{len(all_models)+1}. All models")
            model_choice = input(f"Which model(s) do you want to evaluate? (Enter number or comma-separated list, or {len(all_models)+1} for all): ").strip()
            if model_choice == str(len(all_models)+1):
                to_eval = all_models
            else:
                try:
                    indices = [int(i.strip())-1 for i in model_choice.split(",")]
                    to_eval = [all_models[i] for i in indices if 0 <= i < len(all_models)]
                except:
                    print("Invalid selection. Exiting.")
                    return
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            print("\n--- Evaluating on pan12-test dataset ---")
            test_convs = load_test_convs(PAN12_TEST_PATH, test_50)
            if test_convs:
                # Build settings_str for each model to check for file collisions
                settings_str = f"temp{temperature}_top{top_p}{'_short' if len(test_convs) < 100 else ''}"
                settings_map = {m: settings_str for m in to_eval}
                if not check_existing_result_folders(results_dir, to_eval, settings_map):
                    return
                import subprocess
                import time
                max_parallel = 2
                running = []  # List of (model_name, process)
                to_eval_queue = list(to_eval)
                while to_eval_queue or running:
                    # Start new evaluations if slots available
                    while len(running) < max_parallel and to_eval_queue:
                        m = to_eval_queue.pop(0)
                        args = [
                            'python', __file__, '--eval-worker', m, MODELS_DIR, results_dir,
                            str(temperature), str(top_p)
                        ]
                        proc = subprocess.Popen(args)
                        running.append((m, proc))
                        print(f"Started evaluation for {m} (PID: {proc.pid})")
                    # Check for finished processes
                    for idx in range(len(running)-1, -1, -1):
                        m, proc = running[idx]
                        ret = proc.poll()
                        if ret is not None:
                            print(f"Evaluation finished for {m} (PID: {proc.pid}, exit code: {ret})")
                            running.pop(idx)
                    time.sleep(2)
                print("All evaluations complete.")
            else:
                print("No test data available for evaluation.")
    elif mode_choice == "3":
        # Option to delete models
        print("\nDo you want to delete any fine-tuned models? (y/n)")
        if input().strip().lower() == "y":
            for d in os.listdir(MODELS_DIR):
                d_path = os.path.join(MODELS_DIR, d)
                if d.endswith("_finetuned") and os.path.isdir(d_path):
                    print(f"Delete {d}? (y/n)")
                    if input().strip().lower() == "y":
                        import shutil
                        shutil.rmtree(d_path)
                        print(f"Deleted {d}")
    else:
        print("Invalid mode selection. Exiting.")

if __name__ == "__main__":
    import sys
    if '--eval-worker' in sys.argv:
        # Called as subprocess for parallel evaluation
        _, _, model_name, models_dir, results_dir, temperature, top_p = sys.argv
        from data_loader import load_test_convs
        from Globals import PAN12_TEST_PATH
        test_convs = load_test_convs(PAN12_TEST_PATH)
        eval_worker(model_name, models_dir, results_dir, test_convs, float(temperature), float(top_p))
    elif '--eval-worker-continue' in sys.argv:
        # Called as subprocess for parallel continued evaluation
        _, _, model_name, models_dir, results_dir, temperature, top_p, append_flag = sys.argv
        from data_loader import load_test_convs
        from Globals import PAN12_TEST_PATH
        from utils import get_already_evaluated_conversation_ids
        test_convs = load_test_convs(PAN12_TEST_PATH)
        model_results_dir = os.path.join(results_dir, model_name)
        settings_str = f"temp{temperature}_top{top_p}{'_short' if len(test_convs) < 100 else ''}"
        results_file = os.path.join(model_results_dir, f"{model_name}_{settings_str}.txt")
        already_done = get_already_evaluated_conversation_ids(results_file)
        remaining_convs = {cid: msgs for cid, msgs in test_convs.items() if cid not in already_done}
        def eval_worker_continue(model_name, models_dir, results_dir, test_convs, temperature, top_p, append=False):
            import os
            from evaluation import evaluate_model
            model_path = os.path.join(models_dir, model_name)
            model_results_dir = os.path.join(results_dir, model_name)
            os.makedirs(model_results_dir, exist_ok=True)
            settings_str = f"temp{temperature}_top{top_p}{'_short' if len(test_convs) < 100 else ''}"
            results_file = os.path.join(model_results_dir, f"{model_name}_{settings_str}.txt")
            mode = 'a' if append else 'w'
            with open(results_file, mode, encoding='utf-8') as f:
                if not append:
                    f.write('')
            evaluate_model(model_path, test_convs, results_file=results_file, temperature=temperature, top_p=top_p)
            print(f"Evaluation metrics and outputs written to {results_file}")
        eval_worker_continue(model_name, models_dir, results_dir, remaining_convs, float(temperature), float(top_p), append=(append_flag=='append'))
    else:
        main()
