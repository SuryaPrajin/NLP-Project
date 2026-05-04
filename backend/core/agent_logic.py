class AgentLogic:
    """
    Core state machine for the AI agent behavior.
    """
    def resolve_response(self, intent: str, context: str, user_name: str = None):
        """
        Decides the final response strategy.
        """
        if intent == "greeting":
            greeting = f"Hello {user_name}! " if user_name else "Hello! "
            return {
                "message": f"{greeting}How can I assist you with your orders, refunds, or account today?",
                "action": "null",
                "parameters": {}
            }
        
        if intent == "unknown":
            return {
                "message": "I'm not exactly sure how to help with that. Could you please provide more details about your issue (e.g., an order number or specific concern)?",
                "action": "null",
                "parameters": {}
            }
            
        # For other intents, we rely on the grounded context
        if not context or "No policies available" in context:
            return {
                "message": "I apologize, but I don't have enough specific information in my current guidelines to resolve this. I am escalating your case to our support specialist.",
                "action": "create_support_ticket",
                "parameters": {"issue_type": intent}
            }

        return None # Proceed to LLM/Mock generation

# Singleton
agent_logic = AgentLogic()
