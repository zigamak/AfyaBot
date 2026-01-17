from handlers.base_handler import BaseHandler
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GreetingHandler(BaseHandler):
    """Greeting handler for BEDC Support Bot - redirects to LLM-powered AI chat."""

    def handle_greeting_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict[str, Any]:
        """Handle greeting state messages - redirect to AI chat."""
        logger.info(f"GreetingHandler: Redirecting to AI chat for session {session_id}")
        return self._redirect_to_ai_chat(state, session_id, original_message)

    def generate_initial_greeting(self, state: Dict, session_id: str, user_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate initial greeting and redirect to AI chat."""
        logger.info(f"Session {session_id}: Generating initial greeting for '{user_name}'")
        return self._redirect_to_ai_chat(state, session_id)

    def handle_back_to_main(self, state: Dict, session_id: str, message: str = "") -> Dict[str, Any]:
        """Handle back to main navigation - redirect to AI chat."""
        logger.info(f"Session {session_id}: Returning to main")
        return self._redirect_to_ai_chat(state, session_id, message)

    def _redirect_to_ai_chat(self, state: Dict, session_id: str, user_message: str = "") -> Dict[str, Any]:
        """Redirect user to LLM-powered AI chat with proper state management."""
        
        # Set up state for AI chat
        state["current_state"] = "ai_chat"
        state["current_handler"] = "ai_handler"
        state["conversation_history"] = []
        
        # Preserve user info
        if not state.get("user_name"):
            state["user_name"] = "Customer"
        if not state.get("phone_number"):
            state["phone_number"] = session_id
        
        # Clear any temporary states
        if "fault_data" in state:
            del state["fault_data"]
        if "billing_checked" in state:
            del state["billing_checked"]
            
        self.session_manager.update_session_state(session_id, state)
        
        # Return redirect instruction with the user's message if any
        return {
            "redirect": "ai_handler", 
            "redirect_message": user_message if user_message else "initial_greeting",
            "additional_message": None
        }