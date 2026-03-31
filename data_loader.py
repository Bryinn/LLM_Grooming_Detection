def load_pan12_test_with_labels(json_path, max_records=100000):
    """Load pan12-test JSON and return a DataFrame with all messages and is_predatory label per message."""
    data = []
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
        for conv in j.get('conversations', [])[:max_records]:
            conv_id = conv.get('conversation_id')
            is_pred = conv.get('is_predatory')
            for msg in conv.get('messages', []):
                data.append({
                    'conversation_id': conv_id,
                    'author': msg.get('author'),
                    'text': msg.get('text'),
                    'timestamp': msg.get('timestamp'),
                    'is_predatory': is_pred
                })
    import pandas as pd
    return pd.DataFrame(data)
def load_pan12_test_ground_truth(json_path, max_records=100000):
    """Load pan12-test JSON and return a DataFrame with conversation_id and is_predatory only."""
    data = []
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
        for conv in j.get('conversations', [])[:max_records]:
            data.append({
                'conversation_id': conv.get('conversation_id'),
                'is_predatory': conv.get('is_predatory')
            })
    import pandas as pd
    return pd.DataFrame(data)
def load_pan12_test(json_path, max_records=100000):
    """Load pan12-test JSON (if not too large) and return a DataFrame."""
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
        for conv in j.get('conversations', [])[:max_records]:
            conv_id = conv.get('conversation_id')
            for msg in conv.get('messages', []):
                data.append({
                    'conversation_id': conv_id,
                    'author': msg.get('author'),
                    'text': msg.get('text'),
                    'timestamp': msg.get('timestamp'),
                })
    return pd.DataFrame(data)
import os
import json
import glob
import pandas as pd

def load_pj_dataset(pj_dir):
    """Load all PJ JSON files and return a DataFrame with all conversations and their messages."""
    conversations = []
    for file in glob.glob(os.path.join(pj_dir, '*.json')):
        with open(file, 'r', encoding='utf-8') as f:
            j = json.load(f)
            name = j.get('metadata', {}).get('name', os.path.basename(file))
            contextual_lines = j.get('metadata', {}).get('contextual_lines', [])
            for conv in j.get('conversations', []):
                is_predatory = True  # Always boolean
                data = []
                for msg in conv.get('messages', []):
                    data.append({
                        'author': msg.get('author'),
                        'text': msg.get('text'),
                        'line_number': msg.get('line_number'),
                        'timestamp': msg.get('timestamp'),
                    })
                conversations.append({
                    'conversation_id': conv.get('conversation_id'),
                    'is_predatory': is_predatory,
                    'name': name,
                    'contextual_lines': contextual_lines,
                    'messages': data
                })
    return pd.DataFrame(conversations)

def load_pan12_training(json_path, max_records=0):
    """Load pan12-training JSON and return a DataFrame with all conversations and their messages."""
    conversations = []
    if max_records is None or max_records <= 0:
        max_records = 1000000  # Large default
    with open(json_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
        for conv in j.get('conversations', [])[:max_records]:
            data = []
            for msg in conv.get('messages', []):
                data.append({
                    'author': msg.get('author'),
                    'text': msg.get('text'),
                    'timestamp': msg.get('timestamp'),
                })
            conversations.append({
                'conversation_id': conv.get('conversation_id'),
                'is_predatory': conv.get('is_predatory'),
                'messages': data
            })
    return pd.DataFrame(conversations)

def load_test_convs(test_path, test_50=False):
    try:
        test_df = load_pan12_test(test_path)
        print("Loaded test rows:", len(test_df))
        if test_50:
            first_50_ids = test_df['conversation_id'].drop_duplicates().iloc[:50]
            test_df = test_df[test_df['conversation_id'].isin(first_50_ids)]
            print(f"Using only the first 50 conversations (IDs: {list(first_50_ids)})")
        test_convs = test_df.groupby('conversation_id')['text'].apply(list).to_dict()
        return test_convs
    except Exception as e:
        print(f"Could not load pan12-test: {e}")
        return {}