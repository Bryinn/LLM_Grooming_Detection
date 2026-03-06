import ast
import os
from data_loader import load_pj_dataset, load_pan12_training, load_pan12_test
from multiprocessing import Process
from evaluation import evaluate_companionv1, run_evaluation_summarizer
from training import train_hf_llm
from utils import write_accuracy_summary, prepare_text_data

from Globals import MODEL_IDS, PAN12_TRAIN_PATH, PAN12_TEST_PATH, PJ_DIR
MODELS_DIR = 'models'

model_ids = MODEL_IDS
# --- Multiprocessing worker for evaluation ---
def eval_worker(model_name, models_dir, results_dir, test_convs, initial_prompt, temperature, top_p):
    import os
    from evaluation import evaluate_model
    from utils import write_accuracy_summary
    model_path = os.path.join(models_dir, model_name)
    model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    settings_str = f"temp{temperature}_top{top_p}"
    results_file = os.path.join(model_results_dir, f"{model_name}_{settings_str}.txt")
    summary_file = os.path.join(model_results_dir, f"{model_name}_summary_{settings_str}.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('')
    evaluate_model(model_path, test_convs, initial_prompt=initial_prompt, results_file=results_file, temperature=temperature, top_p=top_p)
    print(f"Evaluation metrics and outputs written to {results_file}")
    write_accuracy_summary(results_file, summary_file)
    print(f"Summary written to {summary_file}")

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
        "You must answer in JSON format: {conversation_id, is_predatory, reasoning}.\n"
        "Given the following conversation, respond strictly in this JSON format.\n"
        "Example 1:\nCONVERSATION:\nHi, how old are you?\nI'm 13.\n### RESPONSE:\n{\"conversation_id\": 1, \"is_predatory\": true, \"reasoning\": \"The user is asking for age and the respondent is a minor.\"}\n"
        "Example 2:\nCONVERSATION:\nHey, want to play a game?\nSure!\n### RESPONSE:\n{\"conversation_id\": 2, \"is_predatory\": false, \"reasoning\": \"The conversation is friendly and does not contain predatory behavior.\"}\n"
        "[/INST]"
    )
    if mode_choice == "1":
        # Ask for dataset size only if training
        print("Select dataset size for training:")
        print("1. Development (first 1000 examples)")
        print("2. Full dataset")
        size_choice = input("Enter 1 or 2: ").strip()
        use_dev = size_choice == "1"
        # Load full conversation dicts from PAN12 training JSON
        import json
        with open(PAN12_TRAIN_PATH, 'r', encoding='utf-8') as f:
            pan12_data = json.load(f)
        conversations = pan12_data['conversations']
        if use_dev:
            conversations = conversations[:dev_limit]
        
        # User interaction
        print("Which models do you want to train?")
        for idx, model_id in enumerate(MODEL_ID_LIST, 1):
            print(f"{idx}. {model_id}")
        print(f"{len(MODEL_ID_LIST)+1}. All")
        model_choice = input(f"Enter 1-{len(MODEL_ID_LIST)} or {len(MODEL_ID_LIST)+1} for all: ").strip()
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
        selected_models = []
        if model_choice == str(len(MODEL_ID_LIST)+1):
            selected_models = MODEL_ID_LIST
        elif model_choice.isdigit() and 1 <= int(model_choice) <= len(MODEL_ID_LIST):
            selected_models = [MODEL_ID_LIST[int(model_choice)-1]]
        else:
            print("Invalid selection.")
            return
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
        mode_choice = int(input("Select option: ").strip())
        if mode_choice == 2:
            run_evaluation_summarizer("results")
            return
        elif mode_choice != 1:
            print("Invalid selection.")
            return
        else:
            try:
                temperature = float(input("Set temperature 0-1 (default 1.0, higher=more creative): ") or "1.0")
                top_p = float(input("Set top_p 0-1 (default 1.0, lower=more focused): ") or "1.0")
            except Exception:
                print("Invalid input. Using default values: temperature=1.0, top_p=1.0")
                temperature = 1.0
                top_p = 1.0
            # Evaluation
            all_models = [d for d in os.listdir(MODELS_DIR) if d.endswith("_finetuned") and os.path.isdir(os.path.join(MODELS_DIR, d))]
            if not all_models:
                print("No fine-tuned models found in the models directory.")
                return
            print("\nAvailable fine-tuned models:")
            for idx, m in enumerate(all_models, 1):
                # Try to match model_id from folder name
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
            # Ask about results file
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            print("\n--- Evaluating on pan12-test dataset ---")
            try:
                test_df = load_pan12_test(PAN12_TEST_PATH)
                print("Loaded test rows:", len(test_df))
                print(test_df.head())
                test_convs = test_df.groupby('conversation_id')['text'].apply(list).to_dict()
                print("Sample test_convs:", list(test_convs.items())[:3])
                # Print a sample prompt for the first conversation
                if test_convs:
                    _, sample_msgs = next(iter(test_convs.items()))
                    conv_text = "\n".join(sample_msgs)
                    prompt = (
                        "[INST] You are a helpful assistant trained to provide sound judgement on predatory conversation. "
                        "You must answer in JSON format: {conversation_id, is_predatory, reasoning}. "
                        "Given the following conversation, respond strictly in this JSON format."
                        "[/INST]"
                    ) + f"\nCONVERSATION:\n{conv_text}\nRespond in JSON: {{conversation_id, is_predatory, reasoning}}"
                    print("Sample prompt (first conversation):", prompt[:500], "...\nLength:", len(prompt))
            except Exception as e:
                print(f"Could not load pan12-test: {e}")
                test_convs = {}
            if test_convs:
                procs = []
                for m in to_eval:
                    p = Process(target=eval_worker, args=(m, MODELS_DIR, results_dir, test_convs, initial_prompt, temperature, top_p))
                    p.start()
                    procs.append(p)
                for p in procs:
                    p.join()

                #evaluate_companionv1(test_convs, initial_prompt=initial_prompt, results_file=None)
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
