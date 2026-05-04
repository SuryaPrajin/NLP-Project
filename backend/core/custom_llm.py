import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import re

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # Linear projections and split into heads
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Concatenate heads and final linear layer
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        return self.out_linear(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm_1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout(self.ff(x2))
        return x

class CustomLegalLLM(nn.Module):
    def __init__(self, d_model=512, n_layers=4, n_heads=8):
        super().__init__()
        from backend.core.bpe_tokenizer import bpe_tokenizer
        bpe_tokenizer.load()
        self.vocab_size = bpe_tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, self.vocab_size)
        self.weights_path = "backend/core/model_weights.pt"
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        pos = self.pos_encoding[:, :seq_len, :]
        x = self.embedding(x) + pos
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

    def save_model(self):
        torch.save(self.state_dict(), self.weights_path)
        print(f"DEBUG: Model saved to {self.weights_path}")

    def load_model(self):
        if os.path.exists(self.weights_path):
            try:
                self.load_state_dict(torch.load(self.weights_path, map_location='cpu'))
                print(f"DEBUG: Model weights loaded from {self.weights_path}")
                return True
            except:
                print("DEBUG: Weights incompatible (likely due to tokenizer change). Starting fresh.")
                return False
        return False

    def generate(self, prompt_text, context_str, max_new_tokens=100, temperature=1.0, top_k=50):
        """
        Generates text using Top-K sampling. 
        If trained weights exist, it produces learned output.
        """
        from backend.core.bpe_tokenizer import bpe_tokenizer
        bpe_tokenizer.load()
        has_weights = self.load_model()
        
        if not has_weights:
            return self._synthesize_response(context_str)
            
        self.eval()
        idx = torch.tensor([bpe_tokenizer.encode(prompt_text)], dtype=torch.long)
        
        try:
            for _ in range(max_new_tokens):
                # Crop to context window
                idx_cond = idx[:, -256:]
                logits = self(idx_cond)
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                
                # Numerical stability: Clip extreme values
                logits = torch.clamp(logits, -100, 100)
                
                # Top-K sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                
                # Final check for validity
                if torch.isnan(probs).any() or (probs <= 0).all():
                    break
                    
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                
                if idx_next.item() == 10: # \n
                    break
            
            generated_answer = bpe_tokenizer.decode(idx[0].tolist())
            return self._synthesize_response(context_str, generated_answer, prompt_text=prompt_text)
        except Exception as e:
            print(f"DEBUG: Generation failed ({e}), falling back to synthesis.")
            return self._synthesize_response(context_str, prompt_text=prompt_text)

    def _synthesize_response(self, context_str, model_draft=None, prompt_text=""):
        """
        Force Structured Response: Answer, Relevant Law, Explanation.
        Always includes mandatory legal disclaimer.
        """
        import json
        try:
            snippets = json.loads(context_str)
        except:
            snippets = []
            
        if not snippets or (isinstance(snippets, str) and "No relevant legal context" in snippets):
            return {
                "query_type": ["UNKNOWN"],
                "answer": "I could not find specific legal provisions in my immediate database to answer this accurately.\n\n**Recommendation:** Please consult a qualified advocate for this specific matter.",
                "disclaimer": "This is for informational purposes only and not legal advice."
            }

        primary = snippets[0]
        act = primary.get('act', 'Criminal Law')
        section = primary.get('section', 'N/A')
        title = primary.get('title', 'Legal Provision')
        content = primary.get('content', '')
        
        # Elite Version: Translate Law -> Human Understanding
        short_title = title.split('—')[-1].strip() if '—' in title else title
        is_weak_draft = not model_draft or len(model_draft) < 20 or model_draft.lower().strip() in prompt_text.lower()
        
        if is_weak_draft:
            # Heuristic-based "translation"
            human_explanation = f"Section {section} of the {act} ({title}) states that "
            
            # Extract punishment text
            punishment_match = re.search(r'(punished with .*?)(?=\n\d+\.|$)', content, re.DOTALL | re.IGNORECASE)
            if punishment_match:
                p_text = punishment_match.group(1).strip().replace('\n', ' ')
                human_explanation += f"a person convicted under this section can be {p_text}. "
            
            # Extract definition/meaning
            definition_match = re.search(r'(Whoever .*?)(?=\n\d+\.|$)', content, re.DOTALL | re.IGNORECASE)
            if definition_match:
                d_text = definition_match.group(1).strip()[:300]
                human_explanation += f"\n\nIn simple language, this provision generally applies when {d_text.lower()}... This law is designed to maintain order and ensure justice in matters involving {short_title.lower()}."
            
            explanation = human_explanation
        else:
            explanation = model_draft
            
        # Final cleanup
        if len(explanation) > 1200:
            explanation = explanation[:1200] + "..."

        # Elite Output Structure
        answer_text = f"⚖️ RELEVANT LAW:\n{act} Section {section} – {title}\n\n"
        answer_text += f"📝 EXPLANATION:\n{explanation}\n\n"
        answer_text += f"🔍 IN YOUR SITUATION:\nIf the act involves intentional elements of {short_title.lower()}, this section may apply. However, the exact applicability depends on factors such as intent, circumstances, and available evidence.\n\n"
        answer_text += "💡 WHAT YOU CAN DO:\n"
        answer_text += f"• Verify: Review whether the act meets the legal definition of {short_title.lower()} under {act}.\n"
        answer_text += "• Consult: Speak with a criminal lawyer for case-specific advice.\n"
        answer_text += "• Prepare: Collect all evidence, witness statements, and related documents."
        
        return {
            "query_type": ["LEGAL_ADVICE"],
            "relevant_sections": [f"Section {section} {act}"],
            "extracted_issues": [title],
            "answer": answer_text,
            "disclaimer": "⚠️ DISCLAIMER:\nThis information is for educational purposes only and does not constitute legal advice."
        }

if __name__ == "__main__":
    # Test initialization
    model = CustomLegalLLM()
    print("Custom Transformer LLM initialized successfully.")
    test_input = torch.randint(0, 5000, (1, 10))
    output = model(test_input)
    print(f"Test Forward Pass Output Shape: {output.shape}")
