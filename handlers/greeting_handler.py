from .base_handler import BaseHandler
import logging
from typing import Dict, Any, List, Optional
import sys

# Configure logging with UTF-8 encoding
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
handler.stream.reconfigure(encoding='utf-8')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class GreetingHandler(BaseHandler):
    """Simplified greeting handler for Gynecology Medical AI (AfyaBot) that redirects to AI chat."""

    def handle_greeting_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict[str, Any]:
        """Handle greeting state messages - redirect everything to AI chat."""
        self.logger.info(f"GreetingHandler: Handling message '{message}' in greeting state for session {session_id}. Redirecting to AI chat.")
        return self._redirect_to_ai_chat(state, session_id)

    def generate_initial_greeting(self, state: Dict, session_id: str, user_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate initial greeting and redirect to AI chat."""
        self.logger.info(f"Session {session_id}: Generating initial greeting for user '{user_name}'. Redirecting to AI chat.")
        return self._redirect_to_ai_chat(state, session_id)

    def handle_back_to_main(self, state: Dict, session_id: str, message: str = "") -> Dict[str, Any]:
        """Handle back to main navigation - redirect to AI chat."""
        self.logger.info(f"Session {session_id}: Back to main requested. Redirecting to AI chat.")
        return self._redirect_to_ai_chat(state, session_id, message)

    def _redirect_to_ai_chat(self, state: Dict, session_id: str, additional_message: str = "") -> Dict[str, Any]:
        """Redirect user to AI chat with proper state management."""
        # Set up state for AI chat
        state["current_state"] = "ai_chat"
        state["current_handler"] = "ai_handler"
        state["conversation_history"] = []
        
        # Ensure user info is preserved
        if not state.get("user_name"):
            state["user_name"] = "Guest"
        if not state.get("phone_number"):
            state["phone_number"] = session_id
            
        self.session_manager.update_session_state(session_id, state)
        
        # Return redirect instruction
        return {
            "redirect": "ai_handler", 
            "redirect_message": "initial_greeting",
            "additional_message": additional_message if additional_message else None
        }