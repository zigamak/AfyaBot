import logging
import re
from typing import Dict
from handlers.base_handler import BaseHandler
from datetime import datetime
from services.ai_service import AIService

logger = logging.getLogger(__name__)

class AIHandler(BaseHandler):
    """Handles AI-powered conversational assistance for BEDC Support Bot using LLM."""

    def __init__(self, config, session_manager, data_manager, whatsapp_service):
        super().__init__(config, session_manager, data_manager, whatsapp_service)
        
        self.ai_service = AIService(config, data_manager)
        self.ai_enabled = self.ai_service.ai_enabled
        
        # BEDC branding
        self.company_image_url = "https://example.com/bedc-logo.jpg"
        
        if not self.ai_enabled:
            logger.warning("AIHandler: Running in fallback mode (no LLM)")
        else:
            logger.info("AIHandler: LLM-powered AI Service initialized")

    def handle_ai_chat_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict:
        """Handle ongoing AI chat state."""
        logger.info(f"AIHandler: Processing message for session {session_id}")
        return self._process_user_message(state, session_id, original_message)

    def handle_ai_menu_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict:
        """Handle AI chat selection state."""
        logger.info(f"AIHandler: Message '{message}' in AI menu state for session {session_id}")
        if message in ["ai_chat", "start_ai_chat", "initial_greeting"]:
            return self._handle_ai_chat_start(state, session_id, original_message)
        elif message in ["back_to_main", "menu"]:
            return self.handle_back_to_main(state, session_id)
        else:
            return self._handle_ai_chat_start(state, session_id, original_message)

    def _handle_ai_chat_start(self, state: Dict, session_id: str, user_message: str = None) -> Dict:
        """Handle AI chat start with LLM support - natural greeting."""
        
        # Set up chat state
        state["current_state"] = "ai_chat"
        state["current_handler"] = "ai_handler"
        
        if "conversation_history" not in state:
            state["conversation_history"] = []
        
        user_name = state.get("user_name", "Customer")
        phone_number = state.get("phone_number", session_id)
        
        # Update session state
        state["welcome_sent"] = True
        self.session_manager.update_session_state(session_id, state)
        
        # If user sent a substantive message, process it
        if user_message and user_message.strip() and user_message.lower() not in [
            "ai_chat", "start_ai_chat", "initial_greeting", "menu", "start", "hello", "hi"
        ]:
            logger.info(f"Processing user message on chat start: {user_message[:50]}")
            return self._process_user_message(state, session_id, user_message)
        
        # Send natural greeting (not robotic)
        greeting_message = f"""Hi {user_name}! I'm here to help with your BEDC account. What can I assist you with today?

I can help with:
📋 Billing inquiries
⚡ Prepaid meter applications (for unmetered customers)
🔧 Fault reports
❓ General questions"""
        
        return self.whatsapp_service.create_text_message(session_id, greeting_message)

    def _process_user_message(self, state: Dict, session_id: str, user_message: str) -> Dict:
        """
        Process user message using LLM-powered AI service with:
        - Account validation
        - Billing confirmation modal
        - Fault confirmation
        - Email masking
        """
        phone_number = state.get("phone_number", session_id)
        user_name = state.get("user_name", "Customer")
        
        logger.info(f"Processing message for {phone_number}: {user_message[:100]}")
        
        try:
            # Get conversation history
            conversation_history = state.get("conversation_history", [])
            
            # Check confirmation states
            pending_billing_confirmation = state.get("pending_billing_confirmation", False)
            pending_fault_confirmation = state.get("pending_fault_confirmation", False)
            
            # Get current session state for context
            session_state = {
                "billing_data": state.get("billing_data", {}),
                "fault_data": state.get("fault_data", {}),
                "account_number": state.get("account_number"),
                "billing_checked": state.get("billing_checked", False),
                "pending_billing_confirmation": pending_billing_confirmation,
                "pending_fault_confirmation": pending_fault_confirmation,
                "phone_number": phone_number
            }
            
            logger.info(f"Session state being passed to AI service: {session_state}")
            
            # Log confirmation states for debugging
            if pending_billing_confirmation:
                logger.info(f"[INFO] BILLING CONFIRMATION MODE: Waiting for user to confirm account details")
                logger.info(f"Billing data: {session_state.get('billing_data', {})}")
            
            if pending_fault_confirmation:
                logger.info(f"[INFO] FAULT CONFIRMATION MODE: Waiting for user to confirm fault details")
                logger.info(f"Fault data: {session_state.get('fault_data', {})}")
            
            logger.info(f"Calling AI service with LLM for: {user_message[:100]}")
            
            # Generate AI response (handles all confirmation logic and email masking)
            ai_response, intent, state_update = self.ai_service.generate_response(
                user_message, 
                conversation_history,
                phone_number,
                user_name,
                session_state
            )
            
            logger.info(f"LLM response generated with intent '{intent}': {ai_response[:100]}")
            
            # Update state with any changes from AI service
            for key, value in state_update.items():
                state[key] = value
                logger.info(f"State updated: {key} = {str(value)[:100]}")
            
            # CRITICAL: Save updated state BEFORE adding to conversation history
            self.session_manager.update_session_state(session_id, state)
            
            # Log the actual state to verify
            logger.info(f"Verified state after update: pending_billing={state.get('pending_billing_confirmation', False)}, pending_fault={state.get('pending_fault_confirmation', False)}")
            
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
            
            # Save to data manager (with full email, not masked - masking is only for display)
            self.data_manager.save_conversation(
                phone_number,
                session_id,
                user_message,
                ai_response,
                intent
            )
            
            # Save updated state
            self.session_manager.update_session_state(session_id, state)
            
            logger.info(f"[SUCCESS] Response sent. States: billing_confirmation={state.get('pending_billing_confirmation', False)}, fault_confirmation={state.get('pending_fault_confirmation', False)}")
            
            return self.whatsapp_service.create_text_message(session_id, ai_response)
        
        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {e}", exc_info=True)
            error_message = (
                "I'm having trouble processing your request right now. "
                "Please try again or contact our office at Ring Road, Benin City."
            )
            return self.whatsapp_service.create_text_message(session_id, error_message)