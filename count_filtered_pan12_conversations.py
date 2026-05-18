import os

def count_conversations_in_folder(folder_path):
    import json
    conversation_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict) and 'conversations' in data and isinstance(data['conversations'], list):
                        conversation_count += len(data['conversations'])
                    else:
                        print(f"Unexpected JSON structure in {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return conversation_count

def main():
    base_dir = os.path.join('filtered_datasets', 'pan12-{}')
    for part in ['training', 'test']:
        folder = base_dir.format(part)
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue
        count = count_conversations_in_folder(folder)
        print(f"{part.capitalize()} set: {count} conversations")

if __name__ == "__main__":
    main()
