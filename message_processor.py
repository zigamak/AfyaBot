import logging
from uuid import uuid4
from handlers.greeting_handler import GreetingHandler
from handlers.ai_handler import AIHandler

logger = logging.getLogger(__name__)

class MessageProcessor:
    """Simplified message processor for Gynecology Medical AI (AfyaBot)."""

    def __init__(self, config, session_manager, data_manager, whatsapp_service):
        self.config = config
        self.session_manager = session_manager
        self.data_manager = data_manager
        self.whatsapp_service = whatsapp_service

        # Initialize only the handlers we need for AfyaBot
        self.greeting_handler = GreetingHandler(config, session_manager, data_manager, whatsapp_service)
        self.ai_handler = AIHandler(config, session_manager, data_manager, whatsapp_service)

        logger.info("MessageProcessor initialized for Gynecology Medical AI (AfyaBot).")

    def process_message(self, message_data, session_id, user_name):
        """Main method to process incoming messages for AfyaBot."""
        try:
            # Retrieve session state
            state = self.session_manager.get_session_state(session_id)
            
            # Update session activity
            self.session_manager.update_session_activity(session_id)

            # Handle different message types
            if isinstance(message_data, dict):
                message = message_data.get("text", "")
            else:
                message = message_data

            original_message = message
            message = message.strip().lower() if message else ""

            # Update user info in session state
            self._update_user_info(state, session_id, user_name)
            self.session_manager.update_session_state(session_id, state)

            # Route to appropriate handler
            response = self._route_to_handler(state, message, original_message, session_id, user_name)

            return response

        except Exception as e:
            logger.error(f"Session {session_id}: Error processing message: {e}", exc_info=True)
            # Reset to AI chat on error
            state = self.session_manager.get_session_state(session_id)
            state["current_state"] = "ai_chat"
            state["current_handler"] = "ai_handler"
            self.session_manager.update_session_state(session_id, state)
            return self.whatsapp_service.create_text_message(
                session_id,
                "⚠️ Something went wrong. Let me help you with your women's health questions. What would you like to know?"
            )

    def _update_user_info(self, state, session_id, user_name):
        """Update user information in session state."""
        if user_name and not state.get("user_name"):
            state["user_name"] = user_name
        if not state.get("user_name"):
            state["user_name"] = "Guest"
        state["phone_number"] = session_id

    def _route_to_handler(self, state, message, original_message, session_id, user_name):
        """Route messages to appropriate handlers - simplified for AfyaBot."""
        current_handler_name = state.get("current_handler", "ai_handler")
        current_state = state.get("current_state", "start")
        
        if "user_name" not in state:
            state["user_name"] = user_name or "Guest"

        try:
            # Global commands (include Swahili equivalents)
            swahili_greetings = ["habari", "mambo", "sasa", "jambo"]
            if message in ["menu", "start", "hello", "hi"] or message in swahili_greetings:
                logger.info(f"Session {session_id}: Global command '{message}' detected. Going to AI chat.")
                return self._start_ai_chat(state, session_id, user_name, original_message)
            
            # For new sessions or start state, go directly to AI chat
            if current_state == "start" or not current_handler_name:
                logger.info(f"Session {session_id}: New session detected. Starting AI chat.")
                return self._start_ai_chat(state, session_id, user_name, original_message)
            
            # Route to AI handler for all interactions
            if current_handler_name == "ai_handler":
                if current_state == "ai_chat":
                    return self.ai_handler.handle_ai_chat_state(state, message, original_message, session_id)
                elif current_state == "kit_request_collection":
                    return self.ai_handler.handle_kit_request_collection_state(state, message, original_message, session_id)  # Assuming you add this method to AIHandler
                else:
                    # Default to AI chat for any other state
                    logger.info(f"Session {session_id}: Unhandled AI state '{current_state}'. Defaulting to AI chat.")
                    return self._start_ai_chat(state, session_id, user_name, original_message)
            
            # Fallback - anything else goes to AI chat
            else:
                logger.info(f"Session {session_id}: Unknown handler '{current_handler_name}'. Redirecting to AI chat.")
                return self._start_ai_chat(state, session_id, user_name, original_message)

        except Exception as e:
            logger.error(f"Session {session_id}: Error in message routing: {e}", exc_info=True)
            return self._start_ai_chat(state, session_id, user_name, original_message)

    def _start_ai_chat(self, state, session_id, user_name, original_message=None):
        """Start or restart AI chat for the user."""
        state["current_state"] = "ai_chat"
        state["current_handler"] = "ai_handler"
        state["user_name"] = user_name or "Guest"
        self.session_manager.update_session_state(session_id, state)
        
        logger.info(f"Session {session_id}: Starting AI chat for user '{user_name}' with message '{original_message}'.")
        return self.ai_handler._handle_ai_chat_start(state, session_id, original_message)

    def cleanup_expired_resources(self):
        """Clean up expired sessions."""
        try:
            self.session_manager.cleanup_expired_sessions()
            logger.info("Resource cleanup completed.")
        except Exception as e:
            logger.error(f"Error in resource cleanup: {e}", exc_info=True)