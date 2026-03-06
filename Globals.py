import os
import ast

PJ_DIR = os.path.join('filtered_datasets', 'PJ')
PAN12_TRAIN_PATH = os.path.join('filtered_datasets', 'pan12-training', 'pan12-sexual-predator-identification-training-corpus-2012-05-01.json')
PAN12_TEST_PATH = os.path.join('filtered_datasets', 'pan12-test', 'pan12-sexual-predator-identification-test-corpus-2012-05-17.json')


MODEL_IDS_PATH = os.path.join(os.path.dirname(__file__), 'model_ids.py')
with open(MODEL_IDS_PATH, 'r', encoding='utf-8') as f:
    model_ids = ast.literal_eval(f.read())
MODEL_IDS = model_ids