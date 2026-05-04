import random

class ActionRouter:
    """
    Executes mock backend actions based on structured LLM triggers.
    In production, this would call real APIs (database, SAP, etc.).
    """
    async def execute(self, action: str, parameters: dict):
        if not action or action == "null":
            return None

        print(f"Executing action: {action} with parameters: {parameters}")
        
        if action == "check_order_status":
            order_id = parameters.get("order_id", "Unknown")
            # Mock status check logic
            statuses = ["In Transit", "Delivered", "Processing", "Delayed"]
            return {"status": "success", "data": {"order_id": order_id, "current_status": random.choice(statuses)}}

        elif action == "initiate_refund":
            order_id = parameters.get("order_id", "Unknown")
            # Mock refund initiation
            return {"status": "success", "data": {"order_id": order_id, "message": "Refund initiated successfully."}}

        elif action == "create_support_ticket":
            issue_type = parameters.get("issue_type", "General")
            description = parameters.get("description", "No description provided.")
            # Mock ticket creation
            ticket_id = f"TIC-{random.randint(1000, 9999)}"
            return {"status": "success", "data": {"ticket_id": ticket_id, "message": f"Ticket {ticket_id} created."}}

        return {"status": "error", "message": f"Unknown action: {action}"}

# Singleton
action_router = ActionRouter()
