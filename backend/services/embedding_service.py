import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self):
        # Using a high-quality local model that runs efficiently on CPU
        # 'all-MiniLM-L6-v2' is small (80MB) and very fast.
        self.model_name = 'all-MiniLM-L6-v2'
        print(f"DEBUG: Initializing Local Embedding Model ({self.model_name})...")
        self.model = SentenceTransformer(self.model_name)
        self.cache = {}

    def get_embedding(self, text: str):
        """Generate embedding for a single text chunk with local model and caching."""
        if text in self.cache:
            return self.cache[text]

        try:
            # Generate embedding using local model
            emb = self.model.encode(text).astype('float32')
            self.cache[text] = emb
            return emb
        except Exception as e:
            print(f"DEBUG: Embedding Error: {e}. Falling back to zero-vector.")
            return np.zeros(384).astype('float32') # MiniLM dimension is 384

    def get_query_embedding(self, query: str):
        """Generate embedding for a search query using the same local model."""
        return self.get_embedding(query)

# Singleton
embedding_service = EmbeddingService()

