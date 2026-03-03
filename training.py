import ast
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import os
MODEL_IDS_PATH = os.path.join(os.path.dirname(__file__), 'model_ids.py')
with open(MODEL_IDS_PATH, 'r', encoding='utf-8') as f:
    model_ids = ast.literal_eval(f.read())
def train_hf_llm(texts, model_id, initial_prompt=None, output_dir=None, suffix="", default_dir_name=None, epoch=1, batch_size=1, learning_rate=2e-5):
    """HuggingFace LLM training function."""
    print(f"[INFO] Using HuggingFace for {model_id} training.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    use_fp16 = False  # Set to True to enable mixed precision
    # Graceful degradation 
    batch_size_options = [8, 4, 2, 1] if batch_size == 1 else [batch_size]
    max_length_options = [256, 128, 64, 32]
    tried_settings = []
    success = False
    force_cpu = False
    for bs in batch_size_options:
        for ml in max_length_options:
            model = tokenizer = dataset = dataloader = optimizer = scaler = None
            try:
                print(f"[INFO] Trying batch_size={bs}, max_length={ml}{' [CPU fallback]' if force_cpu else ''}")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                # Ensure pad_token is set
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model = AutoModelForCausalLM.from_pretrained(model_id)
                if tokenizer.pad_token_id is not None and hasattr(model, 'resize_token_embeddings'):
                    model.resize_token_embeddings(len(tokenizer))
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    if hasattr(model.config, 'use_cache'):
                        model.config.use_cache = False
                # Silence tied weights warning
                if hasattr(model.config, 'tie_word_embeddings'):
                    model.config.tie_word_embeddings = False
                device = torch.device('cpu') if force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                if initial_prompt:
                    texts_ = [(initial_prompt or "") + text for text in texts]
                else:
                    texts_ = texts
                class TextDataset(torch.utils.data.Dataset):
                    def __init__(self, texts, tokenizer, max_length=256):
                        self.examples = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')['input_ids']
                    def __len__(self):
                        return self.examples.shape[0]
                    def __getitem__(self, idx):
                        return {'input_ids': self.examples[idx], 'labels': self.examples[idx]}
                dataset = TextDataset(texts_, tokenizer, max_length=ml)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                model.train()
                print("[INFO] Training started...")
                scaler = torch.cuda.amp.GradScaler() if use_fp16 and torch.cuda.is_available() and not force_cpu else None
                for ep in range(epoch):
                    for batch in tqdm(dataloader, desc=f"Epoch {ep+1}/{epoch}"):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        optimizer.zero_grad()
                        if use_fp16 and torch.cuda.is_available() and not force_cpu:
                            with torch.cuda.amp.autocast():
                                outputs = model(**batch)
                                loss = outputs.loss
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(**batch)
                            loss = outputs.loss
                            loss.backward()
                            optimizer.step()
                output_dir_final = output_dir or f"{default_dir_name or model_id.split('/')[-1]}_{suffix}_finetuned"
                print(f"[INFO] Training complete. Saving finetuned model to {output_dir_final}")
                model.save_pretrained(output_dir_final)
                tokenizer.save_pretrained(output_dir_final)
                success = True
                break
            except RuntimeError as e:
                if 'out of memory' in str(e) or 'CUDA error' in str(e):
                    print(f"[WARN] CUDA OOM with batch_size={bs}, max_length={ml}. Trying lower settings...")
                    tried_settings.append((bs, ml))
                    # cleanup
                    try:
                        if model is not None:
                            model.to('cpu')
                        del model, tokenizer, dataset, dataloader, optimizer, scaler
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                    except Exception:
                        print("[WARN] Error during cleanup after OOM. Continuing with next settings.")
                else:
                    raise
            finally:
                # Extra cleanup in case of partial OOM
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        if success:
            break
    if not success:
        print(f"[ERROR] All batch_size/max_length settings failed due to OOM. Tried: {tried_settings}")
        raise RuntimeError("All batch_size/max_length settings failed due to OOM.")
