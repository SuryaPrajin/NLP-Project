import os
import torch
import json
from PyPDF2 import PdfReader
from backend.core.bpe_tokenizer import bpe_tokenizer

class LegalDataset:
    """
    Reads PDFs from the ML directory and prepares training data using BPE.
    """
    def __init__(self, data_dir="ML/Criminal Law", block_size=256, val_split=0.1):
        self.block_size = block_size
        self.raw_text = self._load_data(data_dir)
        
        # Ensure BPE is trained/loaded
        if not bpe_tokenizer.load():
            print("DEBUG: Training BPE on entire corpus...")
            # We train for 100 merges to get a decent legal vocabulary quickly
            bpe_tokenizer.train(self.raw_text, 100)
            
        print("DEBUG: Encoding dataset with BPE...")
        full_data = torch.tensor(bpe_tokenizer.encode(self.raw_text), dtype=torch.long)
        
        # Train/Val split
        n = int(val_split * len(full_data))
        self.train_data = full_data[n:]
        self.val_data = full_data[:n]
        
        print(f"DEBUG: Dataset Split -> Train: {len(self.train_data)} tokens, Val: {len(self.val_data)} tokens")

    def _load_data(self, data_dir):
        """Extracts text from all PDFs in the directory."""
        all_text = ""
        if not os.path.exists(data_dir):
            print(f"WARNING: Data directory {data_dir} not found.")
            return " " # Minimal data for initialization
            
        for file in os.listdir(data_dir):
            if file.endswith(".pdf"):
                print(f"DEBUG: Extracting text from {file}...")
                path = os.path.join(data_dir, file)
                try:
                    reader = PdfReader(path)
                    for page in reader.pages:
                        all_text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"ERROR: Failed to read {file}: {e}")
        
        print(f"DEBUG: Loaded {len(all_text)} characters of training data.")
        return all_text

    def get_batch(self, split, batch_size):
        """Generates a small batch of data for training or validation."""
        data = self.train_data if split == 'train' else self.val_data
        if len(data) <= self.block_size:
            # Fallback if val_data is too small
            data = self.train_data
            
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y


class LegalSFTDataset:
    """
    Handles Supervised Fine-Tuning (SFT) data formatting.
    Converts (Instruction, Context, Answer) into a single tokenized sequence.
    """
    def __init__(self, sft_path="backend/core/legal_sft_data.json", block_size=512):
        self.block_size = block_size
        with open(sft_path, "r") as f:
            self.raw_data = json.load(f)
        
        print(f"DEBUG: Loaded {len(self.raw_data)} SFT samples.")
        self.encoded_samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for item in self.raw_data:
            # Format: ### Instruction: ... ### Context: ... ### Answer: ...
            full_text = f"### Instruction:\n{item['instruction']}\n\n### Context:\n{item['context']}\n\n### Answer:\n{item['answer']}"
            tokens = bpe_tokenizer.encode(full_text)
            
            # We also need to know where the 'Answer' starts for loss masking
            answer_start_text = f"### Instruction:\n{item['instruction']}\n\n### Context:\n{item['context']}\n\n### Answer:\n"
            answer_start_idx = len(bpe_tokenizer.encode(answer_start_text))
            
            samples.append({
                "tokens": torch.tensor(tokens, dtype=torch.long),
                "answer_start": answer_start_idx
            })
        return samples

    def get_batch(self, batch_size):
        """Randomly samples SFT batches with smart cropping."""
        ix = torch.randint(len(self.encoded_samples), (batch_size,))
        
        batch_x = []
        batch_y = []
        batch_masks = []
        
        for i in ix:
            sample = self.encoded_samples[i]
            tokens = sample["tokens"]
            answer_start = sample["answer_start"]
            
            # If total length exceeds block_size, we crop from the BEGINNING
            # to keep the 'Answer' part intact at the end.
            if len(tokens) > self.block_size:
                shift = len(tokens) - self.block_size
                tokens = tokens[shift:]
                answer_start = max(0, answer_start - shift)
            else:
                padding = torch.zeros(self.block_size - len(tokens), dtype=torch.long)
                tokens = torch.cat([tokens, padding])
            
            x = tokens[:-1]
            y = tokens[1:]
            
            # Mask: 1 for tokens we want to predict (the answer), 0 for others
            mask = torch.zeros(self.block_size - 1)
            # Adjust answer_start for the shifted y (predicting the token at answer_start)
            start = max(0, answer_start - 1)
            
            # End of answer is either the end of tokens or where padding starts
            # If we didn't pad, it's just the end of tokens.
            end = min(self.block_size - 1, len(sample["tokens"]) - (len(sample["tokens"]) - len(tokens)) - 1)
            # Actually, simpler: if we padded, the end is the original length
            end = min(self.block_size - 1, len(sample["tokens"]) - 1)
            
            # Let's be more precise:
            # The tokens we want to predict are those in y that correspond to the answer part.
            # In tokens[1:], the answer starts at index (answer_start - 1).
            # It ends at the end of the non-padded tokens.
            valid_len = min(self.block_size, len(sample["tokens"]))
            mask[start : valid_len - 1] = 1
            
            batch_x.append(x)
            batch_y.append(y)
            batch_masks.append(mask)
            
        return torch.stack(batch_x), torch.stack(batch_y), torch.stack(batch_masks)


if __name__ == "__main__":
    # Test loading
    dataset = LegalDataset()
    if len(dataset.raw_text) > 1:
        x, y = dataset.get_batch('train', 4)
        print(f"Dataset X shape: {x.shape}")
        
    sft = LegalSFTDataset()
    x, y, m = sft.get_batch(2)
    print(f"SFT X shape: {x.shape}, Mask sum: {m.sum().item()}")
