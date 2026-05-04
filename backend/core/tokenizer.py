class LegalTokenizer:
    """
    A character-level tokenizer for the from-scratch LLM.
    Reliable, small, and entirely local.
    """
    def __init__(self):
        # We start with a standard set of characters
        self.chars = sorted(list(set(
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;()'-_\" \n"
        )))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s: str):
        """String to list of integers"""
        return [self.stoi.get(c, self.stoi[' ']) for c in s]

    def decode(self, l: list):
        """List of integers to string"""
        return ''.join([self.itos.get(i, ' ') for i in l])

# Singleton
tokenizer = LegalTokenizer()

if __name__ == "__main__":
    test_str = "Section 302: Murder."
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_str}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Vocab Size: {tokenizer.vocab_size}")
