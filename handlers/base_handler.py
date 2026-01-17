import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class BaseHandler:
    """Base class for all message handlers in BEDC WhatsApp Bot."""

    def __init__(self, config, session_manager, data_manager, whatsapp_service):
        self.config = config
        self.session_manager = session_manager
        self.data_manager = data_manager
        self.whatsapp_service = whatsapp_service
        self.logger = logger

    def handle_back_to_main(self, state: Dict, session_id: str, message: str = "") -> Dict:
        """
        Handle returning to main conversation mode.
        Clears temporary state and redirects to conversational AI.
        """
        # Clear any temporary state but preserve user info
        state["current_state"] = "ai_chat"
        state["current_handler"] = "ai_handler"
        
        # Preserve essential user data
        user_name = state.get("user_name", "Customer")
        phone_number = state.get("phone_number", session_id)
        
        # Clear temporary conversation state
        if "fault_data" in state:
            del state["fault_data"]
        if "billing_inquiry" in state:
            del state["billing_inquiry"]
            
        # Reset conversation history for fresh start if needed
        state["conversation_history"] = []
        state["user_name"] = user_name
        state["phone_number"] = phone_number
            
        self.session_manager.update_session_state(session_id, state)
        self.logger.info(f"Session {session_id} returned to AI chat.")
        
        # Return redirect to AI handler
        return {
            "redirect": "ai_handler", 
            "redirect_message": "initial_greeting",
            "additional_message": message if message else "How can I help you with your BEDC services today?"
        }