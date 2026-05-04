from dotenv import load_dotenv
from backend.core.custom_llm import CustomLegalLLM

load_dotenv()

class LLMCircuitBreaker:
    def __init__(self, failure_threshold=20): # Increased for testing
        self.failure_threshold = failure_threshold
        self.failure_count = 0
        self.state = "CLOSED" 

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def is_available(self):
        return self.state == "CLOSED"

class LLMService:
    def __init__(self):
        # Initializing our custom-built LLM architecture
        print("DEBUG: Initializing Custom From-Scratch LLM (Antigravity-Legal)...")
        self.model = CustomLegalLLM()
        self.breaker = LLMCircuitBreaker()

    async def generate_response(self, prompt: str, history: list, context: str, user_profile: dict = None, matched_intents: list = None):
        """
        Generate a high-precision legal response using our custom Transformer architecture.
        """
        if not self.breaker.is_available():
            return {
                "message": "Our system is currently experiencing high load. Please try again shortly.",
                "disclaimer": "Informational purposes only."
            }

        try:
            # Prepare tokens (simplified for this demonstration)
            # In a production scenario, we would use a tokenizer here.
            # For this 'from scratch' build, the model processes the prompt and context
            # through its synthesis engine.
            
            result = self.model.generate(
                prompt_text=prompt,
                context_str=context
            )
            
            self.breaker.record_success()
            
            return {
                "message": result.get("answer"),
                "query_type": result.get("query_type"),
                "relevant_sections": result.get("relevant_sections"),
                "extracted_issues": result.get("extracted_issues"),
                "disclaimer": result.get("disclaimer")
            }
        except Exception as e:
            print(f"LLM Error: {e}")
            self.breaker.record_failure()
            return {
                "message": "I could not find relevant legal provisions in the provided documents.",
                "disclaimer": "This is for informational purposes only and not legal advice."
            }

# Singleton
llm_service = LLMService()

