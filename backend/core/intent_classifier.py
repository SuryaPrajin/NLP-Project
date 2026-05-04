import re

class IntentClassifier:
    """
    Hybrid intent classifier (Rule-based + logic).
    Classifies queries into Definition, Punishment, Procedure, or Rights.
    """
    def __init__(self):
        self.greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        self.intents = {
            "LEGAL_DEFINITION": [r"what is", r"define", r"meaning of", r"definition", r"stands for"],
            "PUNISHMENT": [r"punishment", r"penalty", r"jail", r"imprisonment", r"fine", r"sentence", r"consequence"],
            "PROCEDURAL": [r"how to", r"procedure", r"fir", r"filing", r"complaint", r"step by step", r"process"],
            "RIGHTS": [r"rights", r"legal right", r"arrest", r"police", r"power", r"can they", r"am i allowed"],
            "SITUATION_BASED": [r"someone", r"attacked", r"victim", r"case", r"if someone", r"committed"]
        }

    def classify(self, text: str):
        text_lower = text.lower().strip()
        matched_intents = []
        
        # 1. Greeting Check
        if any(greet == text_lower or text_lower.startswith(greet + " ") for greet in self.greetings):
            matched_intents.append("GREETING")
            
        # 2. Key Action Checks
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    if intent not in matched_intents:
                        matched_intents.append(intent)
        
        if not matched_intents:
            matched_intents.append("GENERAL_QUERY")
            
        return {"intents": matched_intents, "confidence": 1.0 if matched_intents else 0.0}

# Singleton
intent_classifier = IntentClassifier()
