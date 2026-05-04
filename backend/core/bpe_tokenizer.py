import collections
import json
import os

class LegalBPETokenizer:
    """
    Custom Byte Pair Encoding (BPE) Tokenizer for legal text.
    Moves the model from 'reading characters' to 'reading concepts'.
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {} # (id, id) -> id
        self.vocab = {}  # id -> bytes/string
        self.vocab_path = "backend/core/bpe_vocab.json"

    def train(self, text, num_merges):
        """Trains BPE on the provided text."""
        print(f"DEBUG: Training BPE with {num_merges} merges...")
        # Start with byte-level tokens
        tokens = list(text.encode("utf-8"))
        vocab = {i: bytes([i]) for i in range(256)}
        
        merges = {}
        for i in range(num_merges):
            # Count pairs
            stats = collections.defaultdict(int)
            for pair in zip(tokens, tokens[1:]):
                stats[pair] += 1
            
            if not stats: break
            
            # Find the most frequent pair
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Record the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            # Replace the pair in the token list
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == pair:
                    new_tokens.append(idx)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
            
        self.merges = merges
        self.vocab = vocab
        self._save_vocab()

    def _save_vocab(self):
        # Convert tuple keys to strings for JSON
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            "merges": serializable_merges,
            "vocab": {k: v.hex() for k, v in self.vocab.items()}
        }
        with open(self.vocab_path, "w") as f:
            json.dump(data, f)
        print(f"DEBUG: BPE Vocabulary saved to {self.vocab_path}")

    def load(self):
        if not os.path.exists(self.vocab_path):
            return False
        with open(self.vocab_path, "r") as f:
            data = json.load(f)
        
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        self.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        self.vocab_size = len(self.vocab)
        return True

    def encode(self, text):
        """Encodes text to a list of token IDs."""
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = collections.defaultdict(int)
            for pair in zip(tokens, tokens[1:]):
                stats[pair] += 1
            
            # Find the first mergeable pair (highest priority based on training order)
            best_pair = None
            min_rank = float('inf')
            for pair, rank in self.merges.items():
                if pair in stats:
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair
            
            if best_pair is None:
                break
                
            # Perform the merge
            idx = self.merges[best_pair]
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
                    new_tokens.append(idx)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids):
        """Decodes token IDs back to a string."""
        tokens = b"".join(self.vocab.get(idx, b"") for idx in ids)
        return tokens.decode("utf-8", errors="replace")

# Singleton
bpe_tokenizer = LegalBPETokenizer()

if __name__ == "__main__":
    # Test
    tokenizer = LegalBPETokenizer()
    sample_text = "Section 302 of the IPC refers to punishment for murder. IPC is Indian Penal Code."
    tokenizer.train(sample_text, 50)
    encoded = tokenizer.encode("The IPC Section 302")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
