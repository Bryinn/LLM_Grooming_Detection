import os
def get_already_evaluated_conversation_ids(results_file):
    already_done = set()
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = eval(line.strip()) if line.strip().startswith('{') else None
                    if d and 'conversation_id' in d:
                        already_done.add(d['conversation_id'])
                except Exception:
                    continue
    return already_done

def check_existing_result_folders(results_dir, model_names):
    existing = []
    for m in model_names:
        model_results_dir = os.path.join(results_dir, m)
        if os.path.exists(model_results_dir):
            existing.append(model_results_dir)
    if existing:
        print("WARNING: The following result folders already exist and may be overwritten:")
        for folder in existing:
            print("  ", folder)
        resp = input("Do you want to replace the current evaluation results in these folders? (y/N): ").strip().lower()
        if resp != 'y':
            print("Aborting evaluation.")
            return False
    return True

def select_models(model_list, prompt_all='All'):
    print("Which models do you want to use?")
    for idx, model_id in enumerate(model_list, 1):
        print(f"{idx}. {model_id}")
    print(f"{len(model_list)+1}. {prompt_all}")
    model_choice = input(f"Enter 1-{len(model_list)} or {len(model_list)+1} for all: ").strip()
    if model_choice == str(len(model_list)+1):
        return model_list
    elif model_choice.isdigit() and 1 <= int(model_choice) <= len(model_list):
        return [model_list[int(model_choice)-1]]
    else:
        print("Invalid selection.")
        return []
