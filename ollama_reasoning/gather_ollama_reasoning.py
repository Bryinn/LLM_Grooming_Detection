import os
import json
import ollama
from tqdm import tqdm
from typing import List, Dict, Any

# Use the data loader from the main repo
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import load_pan12_training
from Globals import PAN12_TRAIN_PATH

# Configuration
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'vaultbox/qwen3.5-uncensored:4b')

# Input/output paths
PAN12_JSON_PATH = PAN12_TRAIN_PATH
OUTPUT_PATH = 'ollama_reasoning_results.json'

def load_pan12_conversations(json_path: str) -> List[Dict[str, Any]]:
    """
    Loads PAN12 conversations and their labels using the main data loader.
    Returns a list of dicts: {id, conversation, label}
    """
    df = load_pan12_training(json_path)
    conversations = []
    for _, row in df.iterrows():
        conv_id = row['conversation_id']
        label = 'predatory' if row['is_predatory'] else 'non-predatory'
        # Join all messages into a single conversation string
        conversation = '\n'.join([f"{msg['author']}: {msg['text']}" for msg in row['messages']])
        conversations.append({'id': conv_id, 'conversation': conversation, 'label': label})
    return conversations

def build_prompt(conversation: str, label: str) -> str:
    """
    Build a prompt for the Ollama API based on the conversation and its label.
    """
    return f"""
You are an expert forensic analyst specializing in identifying predatory behavior in online conversations. You have been given a conversation between two individuals, and your task is to give your reasoning for as to why the conversation is labeled as {label.upper()}.
Give reasoning as if you were the one to label this conversation, and explain the key factors that led to this classification. Use evidence from the conversation to support your reasoning.
Below is a chat conversation. The label for this conversation is: {label.upper()}.

Conversation:
{conversation}

Please explain, in detail, why this conversation is labeled as {label.upper()}. Focus on the reasoning and evidence from the conversation. If the label is 'non-predatory', explain why it is not considered predatory.
"""


def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Query the Ollama chat API with the given prompt and return the response.
    """
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
    )
    # The ollama package returns a response object with .message.content
    return response.message.content.strip()



def main():
    conversations = load_pan12_conversations(PAN12_JSON_PATH)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for conv in tqdm(conversations, desc="Processing conversations"):
            prompt = build_prompt(conv['conversation'], conv['label'])
            try:
                reasoning = query_ollama(prompt)
            except Exception as e:
                reasoning = f'Error: {e}'
            out_f.write(json.dumps({
                'id': conv['id'],
                'label': conv['label'],
                'reasoning': reasoning
            }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
