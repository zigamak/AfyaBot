import logging
import re
from typing import Dict
from .base_handler import BaseHandler
from datetime import datetime
from services.ai_service import AIService

logger = logging.getLogger(__name__)

class AIHandler(BaseHandler):
    """Handles AI-powered conversational assistance for BEDC Support Bot."""

    def __init__(self, config, session_manager, data_manager, whatsapp_service):
        super().__init__(config, session_manager, data_manager, whatsapp_service)
        
        self.ai_service = AIService(config, data_manager)
        self.ai_enabled = self.ai_service.ai_enabled
        
        # BEDC branding image (placeholder)
        self.company_image_url = "https://example.com/bedc-logo.jpg"
        
        if not self.ai_enabled:
            logger.warning("AIHandler: AI features disabled as AIService could not be initialized.")
        else:
            logger.info("AIHandler: AIService successfully initialized for BEDC Support Bot.")

    def handle_ai_chat_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict:
        """Handle ongoing AI chat state."""
        logger.info(f"AIHandler: Processing AI chat message for session {session_id}")
        return self._process_user_message(state, session_id, original_message)

    def handle_ai_menu_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict:
        """Handle AI chat selection state."""
        logger.info(f"AIHandler: Handling message '{message}' in AI menu state for session {session_id}. Original: '{original_message}'")
        if message == "ai_chat" or message == "start_ai_chat" or message == "initial_greeting":
            return self._handle_ai_chat_start(state, session_id, original_message)
        elif message == "back_to_main" or message == "menu":
            return self.handle_back_to_main(state, session_id)
        else:
            return self._handle_ai_chat_start(state, session_id, original_message)

    def _handle_ai_chat_start(self, state: Dict, session_id: str, user_message: str = None) -> Dict:
        """Handle AI chat start, setting state and processing user messages or greetings."""
        
        # Set up the chat state
        state["current_state"] = "ai_chat"
        state["current_handler"] = "ai_handler"
        
        # Initialize conversation history if not exists
        if "conversation_history" not in state:
            state["conversation_history"] = []
        
        user_name = state.get("user_name", "Customer")
        
        # Check if welcome message needs to be sent
        welcome_needed = not state.get("welcome_sent", False)
        
        # Build the welcome message
        welcome_message = f"""Hello {user_name}! Welcome to BEDC Customer Support. 🌟

I'm here to help you with:
📋 Billing inquiries and complaints
⚡ Meter applications (MAP enrollment)
🔧 Fault reporting and power outages
❓ General questions about our services

How may I assist you today?"""
        
        # Update session state
        state["welcome_sent"] = True
        self.session_manager.update_session_state(session_id, state)
        
        # If user sent a message with their greeting, process it
        if user_message and user_message.strip() and user_message.lower() not in [
            "ai_chat", "start_ai_chat", "initial_greeting", "menu", "start"
        ]:
            logger.info(f"Processing user message on chat start: {user_message[:50]}")
            return self._process_user_message(state, session_id, user_message)
        else:
            return self.whatsapp_service.create_text_message(session_id, welcome_message)

    def _process_user_message(self, state: Dict, session_id: str, user_message: str) -> Dict:
        """Process user message using AI service."""
        phone_number = state.get("phone_number", session_id)
        user_name = state.get("user_name", "Customer")
        
        logger.info(f"Processing message for {phone_number}: {user_message[:100] if user_message else 'None'}")
        
        if not self.ai_service.ai_enabled:
            error_message = (
                "Sorry, the AI chat feature is currently unavailable. "
                "Please contact our office directly at Ring Road, Benin City."
            )
            return self.whatsapp_service.create_text_message(session_id, error_message)
        
        try:
            # Get conversation history
            conversation_history = state.get("conversation_history", [])
            
            # Get current session state for context
            session_state = {
                "fault_data": state.get("fault_data", {}),
                "account_number": state.get("account_number"),
                "needs_account_number": state.get("needs_account_number", False)
            }
            
            logger.info(f"Calling AI service for message: {user_message[:100]}")
            
            # Generate AI response
            ai_response, intent, state_update = self.ai_service.generate_response(
                user_message, 
                conversation_history,
                phone_number,
                user_name,
                session_state
            )
            
            logger.info(f"AI response generated with intent '{intent}': {ai_response[:100] if ai_response else 'None'}")
            
            # Update state with any changes from AI service
            for key, value in state_update.items():
                state[key] = value
            
            # Update conversation history
            conversation_entry = {
                "user": user_message,
                "assistant": ai_response,
                "intent": intent,
                "timestamp": datetime.now().isoformat()
            }
            conversation_history.append(conversation_entry)
            
            # Keep only last 20 exchanges
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
            
            state["conversation_history"] = conversation_history
            
            # Save to data manager
            self.data_manager.save_conversation(
                phone_number,
                session_id,
                user_message,
                ai_response,
                intent
            )
            
            self.session_manager.update_session_state(session_id, state)
            
            return self.whatsapp_service.create_text_message(session_id, ai_response)
        
        except Exception as e:
            logger.error(f"Error processing user message for session {session_id}: {e}", exc_info=True)
            error_message = (
                "I'm having trouble processing your request right now. "
                "Please try again or contact our office directly."
            )
            return self.whatsapp_service.create_text_message(session_id, error_message)