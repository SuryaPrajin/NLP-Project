import torch
import torch.optim as optim
from backend.core.custom_llm import CustomLegalLLM
from backend.core.dataset_loader import LegalDataset
import time

def train_model(epochs=100, batch_size=12, lr=3e-4, eval_iters=10):
    """
    Main training loop with validation tracking.
    """
    print("--- Initializing Optimized Training Pipeline ---")
    dataset = LegalDataset()
    model = CustomLegalLLM()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    start_time = time.time()

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = dataset.get_batch(split, batch_size)
                logits = model(X)
                B, T, C = logits.shape
                loss = criterion(logits.view(B*T, C), Y.view(B*T))
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    print(f"DEBUG: Starting optimized training for {epochs} iterations...")
    for i in range(epochs):
        # Evaluation
        if i % (epochs // 5) == 0:
            losses = estimate_loss()
            print(f"Step {i}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

        # Batch
        x, y = dataset.get_batch('train', batch_size)
        
        # Forward
        logits = model(x)
        B, T, C = logits.shape
        loss = criterion(logits.view(B*T, C), y.view(B*T))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()
    print(f"--- Training Complete in {end_time - start_time:.2f}s ---")
    model.save_model()


def train_sft(epochs=200, batch_size=8, lr=1e-4):
    """
    Supervised Fine-Tuning (SFT) loop.
    Focuses on teaching the model to follow instructions and ground answers.
    """
    print("--- Starting Supervised Fine-Tuning (SFT) ---")
    from backend.core.dataset_loader import LegalSFTDataset
    dataset = LegalSFTDataset()
    model = CustomLegalLLM()
    model.load_model() # Start from pre-trained weights
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # We use reduction='none' so we can apply our masks manually
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    model.train()
    start_time = time.time()

    for i in range(epochs):
        X, Y, M = dataset.get_batch(batch_size)
        
        logits = model(X)
        B, T, C = logits.shape
        
        # Raw loss per token
        loss_raw = criterion(logits.view(B*T, C), Y.view(B*T))
        loss_raw = loss_raw.view(B, T)
        
        # Apply masks: Only count loss where M == 1 (the answer part)
        loss = (loss_raw * M).sum() / M.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"SFT Step {i}: Masked Loss = {loss.item():.4f}")

    print(f"--- SFT Complete in {time.time() - start_time:.2f}s ---")
    model.save_model()

if __name__ == "__main__":
    # 1. First, ensure the model has basic language patterns
    # train_model(epochs=100) 
    
    # 2. Then, align it with instructions
    train_sft(epochs=100)
