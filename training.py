from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch


def build_sft_pairs(conversations, initial_prompt=None):
    """Build SFT prompt/response pairs from conversation dicts."""
    pairs = []
    def generate_reasoning(conv_text, is_pred):
        """
        Generate a verbose, context-aware reason for the label.
        This version summarizes the conversation, highlights key behaviors, and provides a rationale.
        """
        import re
        text = conv_text.strip()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        summary = []
        predatory_signs = []
        non_predatory_signs = []
        # Analyze for explicit, secrecy, age, off-platform, grooming, and general tone
        for line in lines:
            l = line.lower()
            if any(w in l for w in ["sex", "nude", "naked", "dick", "pussy", "boobs", "cum", "orgasm", "masturbate", "suck", "blowjob"]):
                predatory_signs.append(f"Explicit/sexual language: '{line}'")
            if any(w in l for w in ["don't tell", "dont tell", "secret", "keep this between us", "private", "hide", "discreet"]):
                predatory_signs.append(f"Secrecy or privacy request: '{line}'")
            if any(w in l for w in ["how old", "years old", "i am ", "i'm ", "age"]):
                predatory_signs.append(f"Age-related inquiry: '{line}'")
            if any(w in l for w in ["snapchat", "kik", "whatsapp", "phone number", "text me at", "call me at", "add me on"]):
                predatory_signs.append(f"Off-platform move attempt: '{line}'")
            if re.search(r'\b(babe|sweetie|cutie|beautiful|pretty|handsome|love you|miss you)\b', l):
                predatory_signs.append(f"Grooming/affectionate language: '{line}'")
            if any(w in l for w in ["school", "homework", "parents", "teacher"]):
                non_predatory_signs.append(f"Child-appropriate topic: '{line}'")
            if re.search(r'\b(no|stop|not interested|leave me alone|uncomfortable)\b', l):
                non_predatory_signs.append(f"Resistance or discomfort: '{line}'")
        # Summarize conversation
        if len(lines) > 2:
            summary.append(f"The conversation consists of {len(lines)} messages.")
            summary.append(f"First message: '{lines[0]}'")
            summary.append(f"Last message: '{lines[-1]}'")
        else:
            summary.append("Short conversation.")
        # Compose reason
        if is_pred:
            reason = "This conversation is labeled as predatory. "
            if predatory_signs:
                reason += "Key indicators include: " + "; ".join(predatory_signs) + ". "
            else:
                reason += "No explicit predatory phrases detected, but the overall context or behavior suggests grooming or inappropriate intent. "
        else:
            reason = "This conversation is labeled as non-predatory. "
            if non_predatory_signs:
                reason += "Notable signs of normalcy: " + "; ".join(non_predatory_signs) + ". "
            else:
                reason += "No clear predatory indicators or concerning behavior were found. "
        reason += " ".join(summary)
        return reason.strip()

    for conv in conversations:
        is_pred = conv.get('is_predatory')
        conv_text = "\n".join([msg['text'] for msg in conv['messages']])
        delimiter = "\n### RESPONSE:\n"
        prompt = (initial_prompt or "") + f"\nCONVERSATION:\n{conv_text}\n\nNow, based only on the above conversation, respond strictly with a single valid JSON object in this format: {{\"is_predatory\": true/false, \"reasoning\": \"...\"}}. Do not include any other text." + delimiter
        reasoning = generate_reasoning(conv_text, is_pred)
        response = f'{{"is_predatory": {str(is_pred).lower()}, "reasoning": "{reasoning}"}}'
        pairs.append((prompt, response))
    return pairs

def train_hf_llm(conversations, model_id, initial_prompt=None, output_dir=None, suffix="", default_dir_name=None, epoch=1, batch_size=1, learning_rate=2e-5):
    """HuggingFace LLM SFT training function."""
    print(f"[INFO] Using HuggingFace for {model_id} SFT training.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    use_fp16 = False  # Set to True to enable mixed precision
    batch_size_options = [8, 4, 2, 1] if batch_size == 1 else [batch_size]
    max_length_options = [256, 128, 64, 32]
    tried_settings = []
    success = False
    force_cpu = False
    sft_pairs = build_sft_pairs(conversations, initial_prompt)
    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, pairs, tokenizer, max_length=256):
            self.input_ids = []
            self.labels = []
            for prompt, response in pairs:
                prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
                response_ids = tokenizer(response, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
                input_ids = prompt_ids + response_ids
                labels = [-100]*len(prompt_ids) + response_ids
                # Truncate if too long
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    labels = labels[:max_length]
                self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                self.labels.append(torch.tensor(labels, dtype=torch.long))
        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, idx):
            return {'input_ids': self.input_ids[idx], 'labels': self.labels[idx]}
    
    # Collate function to pad sequences in the batch
    def sft_collate_fn(batch):
        # Pads input_ids and labels to the max length in the batch
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        max_len = max(x.size(0) for x in input_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids_padded = torch.stack([torch.cat([x, torch.full((max_len - x.size(0),), pad_id, dtype=torch.long)]) for x in input_ids])
        labels_padded = torch.stack([torch.cat([y, torch.full((max_len - y.size(0),), -100, dtype=torch.long)]) for y in labels])
        return {'input_ids': input_ids_padded, 'labels': labels_padded}
    for bs in batch_size_options:
        for ml in max_length_options:
            model = None
            tokenizer = None
            dataset = None
            dataloader = None
            optimizer = None
            scaler = None
            try:
                # Set up the model, tokenizer, dataset, dataloader, optimizer, and scaler with the current batch size and max length settings.
                print(f"[INFO] Trying batch_size={bs}, max_length={ml}{' [CPU fallback]' if force_cpu else ''}")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                model_max_length = getattr(model.config, 'max_position_embeddings', 2048)
                effective_max_length = min(ml, model_max_length)
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                if tokenizer.pad_token_id is not None and hasattr(model, 'resize_token_embeddings'):
                    model.resize_token_embeddings(len(tokenizer))
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    if hasattr(model.config, 'use_cache'):
                        model.config.use_cache = False
                device = torch.device('cpu') if force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                dataset = SFTDataset(sft_pairs, tokenizer, max_length=effective_max_length)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=sft_collate_fn)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                model.train()
                
                # Do theactual training
                print("[INFO] SFT Training started...")
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
                model.save_pretrained(output_dir_final, safe_serialization=True)
                model.save_pretrained(output_dir_final, safe_serialization=False)
                tokenizer.save_pretrained(output_dir_final)
                success = True
                break
            # Fallback to smaller batch size or max_length if OOM occurs
            except RuntimeError as e:
                if 'out of memory' in str(e) or 'CUDA error' in str(e):
                    print(f"[WARN] CUDA OOM with batch_size={bs}, max_length={ml}. Trying lower settings...")
                    tried_settings.append((bs, ml))
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
