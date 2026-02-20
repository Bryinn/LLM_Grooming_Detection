import os
import re
import json
from datetime import datetime
from collections import Counter

import xml.etree.ElementTree as ET

def parse_pan12_xml(xml_path):
    """
    Parse a PAN12 XML file and extract conversations.
    Returns a list of conversations, each as a dict with participants and messages.
    Each message: {'author': ..., 'text': ..., 'timestamp': ..., 'line_number': ...}
    """
    conversations = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # PAN12 format: <conversations><conversation id=...>...</conversation>...</conversations>
        for conv in root.findall('.//conversation'):
            messages = []
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
                messages.append({
                    'author': author,
                    'text': text,
                    'timestamp': ts,
                    'line_number': i+1
                })
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
    for filename in os.listdir(input_dir):
        if not filename.endswith(".xml"):
            continue
        input_path = os.path.join(input_dir, filename)
        conversations = parse_pan12_xml(input_path)
        # Only keep conversations with exactly 2 participants and at least 6 messages
        filtered = []
        for conv in conversations:
            participants = set(m['author'] for m in conv['messages'] if m.get('author'))
            if len(participants) == 2 and len(conv['messages']) >= 6:
                filtered.append(conv)
        filtered_in = len(filtered)
        filtered_out = len(conversations) - filtered_in
        total_in += filtered_in
        total_out += filtered_out
        output_path = os.path.join(output_dir, filename.replace('.xml', '.json'))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'conversations': filtered}, f, indent=2)
        print(f"{filename}: {filtered_in} conversations kept, {filtered_out} filtered out")
    print(f"Summary: {total_in} conversations kept, {total_out} filtered out.")

if __name__ == "__main__":
    filter_pan12_conversations("datasets/pan12-training", "filtered_datasets/pan12-training")
    filter_pan12_conversations("datasets/pan12-test", "filtered_datasets/pan12-test")
