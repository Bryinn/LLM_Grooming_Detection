

import ast
import os
from data_loader import load_pj_dataset, load_pan12_training, load_test_convs
from evaluation import evaluate_companionv1, run_evaluation_summarizer
from training import train_hf_llm
from utils import select_models, check_existing_result_folders, get_already_evaluated_conversation_ids

from Globals import MODEL_IDS, PAN12_TRAIN_PATH, PAN12_TEST_PATH, PJ_DIR
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
    print("3. Delete fine-tuned models")
    mode_choice = input("Enter 1, 2, or 3: ").strip()

    dev_limit = 1000
    # Few-shot examples for more robust output
    initial_prompt = (
        "[INST] You are a helpful assistant trained to provide sound judgement on predatory conversation in the context of child predators targeting children in sexual ways.\n"
        "You must answer in JSON format: {is_predatory, reasoning}.\n"
        "Given the following conversation, respond strictly in this JSON format.\n"
        "[/INST]"
    )
    if mode_choice == "1":
        # Ask for dataset size only if training
        print("Select dataset size for training:")
        print("1. Development (first 1000 examples)")
        print("2. Full dataset")
        size_choice = input("Enter 1 or 2: ").strip()
        use_dev = size_choice == "1"
        # Load full conversations for training
        pan12_df = load_pan12_training(PAN12_TRAIN_PATH, max_records=dev_limit if use_dev else 0)
        #pan12_convs = pan12_df.to_dict(orient='records')
        conversations = pan12_df.to_dict(orient='records')
        #if not use_dev:
        #    pj_df = load_pj_dataset(PJ_DIR)
        #    pj_convs = pj_df.to_dict(orient='records')
        #    conversations = pan12_convs + pj_convs
        #else:
        #    conversations = pan12_convs

        # Model selection
        selected_models = select_models(MODEL_ID_LIST, prompt_all='All')
        if not selected_models:
            return
        # Ask for number of epochs
        try:
            epoch = int(input("Enter number of epochs (e.g. 1-10, default 1): ") or "1")
        except Exception:
            print("Invalid input. Using default: 1 epoch.")
            epoch = 1
        # Ask for learning rate
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
        # Add training params to folder name
        mode_str = "dev" if use_dev else "full"
        for model_id in selected_models:
            output_dir = os.path.join(
                MODELS_DIR,
                f"{model_id.split('/')[-1]}_{mode_str}_ep{epoch}_lr{learning_rate}{suffix}_finetuned"
            )
            train_hf_llm(
                conversations,
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
                print("No fine-tuned models found in the models directory.")
                return
            print("\nAvailable fine-tuned models:")
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
                    # Invoke evaluate_model with append mode for the results file when applicable
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
                    eval_worker_continue(m, MODELS_DIR, results_dir, remaining_convs, temperature, top_p, append=True)
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
                print("No fine-tuned models found in the models directory.")
                return
            print("\nAvailable fine-tuned models:")
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
                for m in to_eval:
                    eval_worker(m, MODELS_DIR, results_dir, test_convs, temperature=temperature, top_p=top_p)
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
    main()
