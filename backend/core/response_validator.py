class ResponseValidator:
    """
    Validates LLM output against safety and business rules.
    """
    def validate(self, llm_output: dict, user_query: str) -> bool:
        """
        Returns True if response is valid, False otherwise.
        """
        # Rule 1: Must have a message
        if not llm_output.get("message") or len(llm_output["message"]) < 5:
            return False

        # Rule 2: Citation check
        msg = llm_output["message"].lower()
        required_keywords = ["ipc", "crpc", "act", "indian penal code", "code of criminal procedure", "indian evidence act"]
        
        if not any(kw in msg for kw in required_keywords):
             # Basic check to see if the bot is citing something
             if "i could not find" not in msg:
                 return False

        return True

# Singleton
response_validator = ResponseValidator()
