import os
import json

import xml.etree.ElementTree as ET

def parse_pan12_xml(xml_path):
    """
    Parse a PAN12 XML file and extract conversations.
    Returns a list of conversations, each as a dict with participants and messages.
    Each message: {'author': ..., 'text': ..., 'timestamp': ..., 'line_number': ...}
    """
    conversations = []
    from datetime import datetime, timedelta
    def parse_time(ts):
        # Try to parse various timestamp formats, fallback to None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                continue
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # PAN12 format: <conversations><conversation id=...>...</conversation>...</conversations>
        for conv in root.findall('.//conversation'):
            raw_messages = []
            participants = set()
            for i, msg in enumerate(conv.findall('./message')):
                # PAN12: <message line="n"><author>...</author><time>...</time><text>...</text></message>
                author_elem = msg.find('author')
                author = author_elem.text.strip() if author_elem is not None and author_elem.text else 'unknown'
                participants.add(author)
                text_elem = msg.find('text')
                text = text_elem.text.strip() if text_elem is not None and text_elem.text else ''
                time_elem = msg.find('time')
                timestamp = time_elem.text.strip() if time_elem is not None and time_elem.text else None
                ts = timestamp if timestamp else 'N/A'
                dt = parse_time(ts) if ts and ts != 'N/A' else None
                raw_messages.append({
                    'author': author,
                    'text': text,
                    'timestamp': ts,
                    'dt': dt,
                    'line_number': i+1
                })
            # Sort by datetime if available, else by line_number
            raw_messages.sort(key=lambda m: m['dt'] if m['dt'] else m['line_number'])
            # Time-based splitting: split if >= 3 hour gap
            split_convs = []
            current = []
            prev_dt = None
            for m in raw_messages:
                if prev_dt and m['dt'] and (m['dt'] - prev_dt).total_seconds() >= 3*3600:
                    if current:
                        split_convs.append(current)
                    current = []
                current.append(m)
                prev_dt = m['dt'] if m['dt'] else prev_dt
            if current:
                split_convs.append(current)
            for messages in split_convs:
                # Remove 'dt' field for output
                for msg in messages:
                    msg.pop('dt', None)
                conversations.append({
                    'participants': list(participants),
                    'messages': messages
                })
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return conversations

def filter_pan12_conversations(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    total_in = 0
    total_out = 0

    # Determine which predator file to use
    if 'training' in input_dir:
        predator_file = os.path.join('datasets', 'pan12-training', 'pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt')
    elif 'test' in input_dir:
        predator_file = os.path.join('datasets', 'pan12-test', 'pan12-sexual-predator-identification-groundtruth-problem1.txt')
    else:
        predator_file = None

    predator_ids = set()
    if predator_file and os.path.exists(predator_file):
        with open(predator_file, 'r', encoding='utf-8') as pf:
            for line in pf:
                line = line.strip()
                if line:
                    predator_ids.add(line)

    # Group conversations by participant pairs across all files
    pair_cases = {}
    file_conversation_counts = {}
    for filename in os.listdir(input_dir):
        if not filename.endswith(".xml"):
            continue
        input_path = os.path.join(input_dir, filename)
        conversations = parse_pan12_xml(input_path)
        file_conversation_counts[filename] = len(conversations)
        for idx, conv in enumerate(conversations):
            participants = set(m['author'] for m in conv['messages'] if m.get('author'))
            # Only keep conversations with exactly 2 participants and at least 6 messages (after time-based split)
            if len(participants) == 2 and len(conv['messages']) >= 6:
                pair = tuple(sorted(participants))
                # Mark as predatory if any participant is in predator_ids
                is_pred = any(p in predator_ids for p in participants)
                case = pair_cases.setdefault(pair, {
                    'participants': list(pair),
                    'messages': [],
                    'is_predatory': is_pred,
                    'conversation_ids': []
                })
                case['is_predatory'] = case['is_predatory'] or is_pred
                case['messages'].extend(conv['messages'])
                case['conversation_ids'].append(f"{filename}:{idx}")
                total_in += 1
            else:
                total_out += 1
    # Write all grouped cases to a single output file
    output_path = os.path.join(output_dir, "grouped_cases.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'cases': list(pair_cases.values())}, f, indent=2)
    print(f"Grouped {total_in} conversations into {len(pair_cases)} cases. {total_out} conversations filtered out.")

if __name__ == "__main__":
    filter_pan12_conversations("datasets/pan12-training", "filtered_datasets/pan12-training")
    filter_pan12_conversations("datasets/pan12-test", "filtered_datasets/pan12-test")
