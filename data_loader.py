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
    """Load all PJ JSON files and return a DataFrame with all messages."""
    data = []
    for file in glob.glob(os.path.join(pj_dir, '*.json')):
        with open(file, 'r', encoding='utf-8') as f:
            j = json.load(f)
            name = j.get('metadata', {}).get('name', os.path.basename(file))
            for conv in j.get('conversations', []):
                for msg in conv.get('messages', []):
                    data.append({
                        'file': name,
                        'author': msg.get('author'),
                        'text': msg.get('text'),
                        'timestamp': msg.get('timestamp'),
                    })
    return pd.DataFrame(data)

def load_pan12_training(json_path, max_records=100000):
    """Load pan12-training JSON (if not too large) and return a DataFrame."""
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
        for conv in j.get('conversations', [])[:max_records]:
            for msg in conv.get('messages', []):
                data.append({
                    'author': msg.get('author'),
                    'text': msg.get('text'),
                    'timestamp': msg.get('timestamp'),
                })
    return pd.DataFrame(data)
