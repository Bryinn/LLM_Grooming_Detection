import ast
import os
from data_loader import load_pj_dataset, load_pan12_training, load_pan12_test
from evaluation import evaluate_companionv1
from training import train_hf_llm
from utils import write_accuracy_summary, prepare_text_data

MODEL_IDS_PATH = os.path.join(os.path.dirname(__file__), 'model_ids.py')
with open(MODEL_IDS_PATH, 'r', encoding='utf-8') as f:
    MODEL_ID_LIST = ast.literal_eval(f.read())

# Unified LLM Training Script
# Models: gpt-oss-20b, compAnIonv1, Qwen 1.5 0.5b
# Data: filtered_datasets/PJ, filtered_datasets/pan12-training


# Paths

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
PJ_DIR = os.path.join('filtered_datasets', 'PJ')
PAN12_TRAIN_PATH = os.path.join('filtered_datasets', 'pan12-training', 'pan12-sexual-predator-identification-training-corpus-2012-05-01.json')
PAN12_TEST_PATH = os.path.join('filtered_datasets', 'pan12-test', 'pan12-sexual-predator-identification-test-corpus-2012-05-17.json')



def main():
    # User interaction: train or evaluate
    print("Select mode:")
    print("1. Train models")
    print("2. Evaluate models")
    print("3. Delete fine-tuned models")
    mode_choice = input("Enter 1, 2, or 3: ").strip()

    dev_limit = 1000
    initial_prompt = (
        "[INST] You are a helpful assistant trained to provide sound judgement on predatory conversation. "
        "You must answer in JSON format: {conversation_id, is_predatory, reasoning}. "
        "Given the following conversation, respond strictly in this JSON format."
        "[/INST]"
    )
    if mode_choice == "1":
        # Ask for dataset size only if training
        print("Select dataset size for training:")
        print("1. Development (first 1000 examples)")
        print("2. Full dataset")
        size_choice = input("Enter 1 or 2: ").strip()
        use_dev = size_choice == "1"
        texts = prepare_text_data(load_pj_dataset, load_pan12_training, PJ_DIR, PAN12_TRAIN_PATH)
        if use_dev:
            texts = texts[:dev_limit]
        print("Which models do you want to train?")
        for idx, model_id in enumerate(MODEL_ID_LIST, 1):
            print(f"{idx}. {model_id}")
        print(f"{len(MODEL_ID_LIST)+1}. All")
        model_choice = input(f"Enter 1-{len(MODEL_ID_LIST)} or {len(MODEL_ID_LIST)+1} for all: ").strip()
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
        for model_id in selected_models:
            output_dir = os.path.join(MODELS_DIR, f"{model_id.split('/')[-1]}_{suffix}_finetuned")
            train_hf_llm(
                texts,
                model_id=model_id,
                initial_prompt=initial_prompt,
                output_dir=output_dir,
                suffix=suffix,
                default_dir_name=model_id.split('/')[-1]
            )
    elif mode_choice == "2":
        try:
            temperature = float(input("Set temperature (default 1.0, higher=more creative): ") or "1.0")
            top_p = float(input("Set top_p (default 0.9, lower=more focused): ") or "0.9")
        except Exception:
            print("Invalid input. Using default values: temperature=1.0, top_p=0.9")
            temperature = 1.0
            top_p = 0.9
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
        add_temp_to_file = input("Add temperature/top_p to results file name? (y/n): ").strip().lower() == "y"
        if add_temp_to_file:
            results_file = os.path.join(results_dir, f"evaluation_results_temp{temperature}_top{top_p}.txt")
            summary_file = os.path.join(results_dir, f"evaluation_summary_temp{temperature}_top{top_p}.txt")
        else:
            results_file = os.path.join(results_dir, "evaluation_results.txt")
            summary_file = os.path.join(results_dir, "evaluation_summary.txt")
        print("\n--- Evaluating on pan12-test dataset ---")
        try:
            test_df = load_pan12_test(PAN12_TEST_PATH)
            print("Loaded test rows:", len(test_df))
            print(test_df.head())
            test_convs = test_df.groupby('conversation_id')['text'].apply(list).to_dict()
            print("Sample test_convs:", list(test_convs.items())[:3])
            # Print a sample prompt for the first conversation
            if test_convs:
                sample_id, sample_msgs = next(iter(test_convs.items()))
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
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write('')
        if test_convs:
            from evaluation import evaluate_model
            for m in to_eval:
                model_path = os.path.join(MODELS_DIR, m)
                evaluate_model(model_path, test_convs, initial_prompt=initial_prompt, results_file=results_file, temperature=temperature, top_p=top_p)
            evaluate_companionv1(test_convs, initial_prompt=initial_prompt, results_file=results_file)
            print(f"Evaluation metrics and outputs written to {results_file}")
            write_accuracy_summary(results_file, summary_file)
            print(f"Summary written to {summary_file}")
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
