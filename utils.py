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

def check_existing_result_folders(results_dir, model_names, settings_strs=None):
    """
    Checks for existing result files for each model and settings string.
    Only warns if a file with the same settings exists for a model.
    settings_strs: Optional dict mapping model_name -> settings_str (filename part)
    """
    existing = []
    for m in model_names:
        model_results_dir = os.path.join(results_dir, m)
        if settings_strs and m in settings_strs:
            # Only check for file with this settings string
            fname = f"{m}_{settings_strs[m]}.txt"
            fpath = os.path.join(model_results_dir, fname)
            if os.path.exists(fpath):
                existing.append(fpath)
        else:
            # Fallback: warn if any .txt file exists in the folder
            if os.path.exists(model_results_dir):
                for f in os.listdir(model_results_dir):
                    if f.endswith('.txt'):
                        existing.append(os.path.join(model_results_dir, f))
    if existing:
        print("WARNING: The following result files already exist and may be overwritten:")
        for f in existing:
            print("  ", f)
        resp = input("Do you want to replace the current evaluation results in these files? (y/N): ").strip().lower()
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
