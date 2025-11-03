import logging
import re
from typing import Dict
from .base_handler import BaseHandler
from datetime import datetime
from services.ai_service import AIService

logger = logging.getLogger(__name__)

class AIHandler(BaseHandler):
    """Handles AI-powered conversational assistance for a gynecology medical chatbot."""

    def __init__(self, config, session_manager, data_manager, whatsapp_service):
        super().__init__(config, session_manager, data_manager, whatsapp_service)
        
        self.ai_service = AIService(config, data_manager)
        self.ai_enabled = self.ai_service.ai_enabled
        
        # Medical branding image (placeholder)
        self.company_image_url = "https://example.com/afyabot-logo.jpg"
        
        if not self.ai_enabled:
            logger.warning("AIHandler: AI features disabled as AIService could not be initialized.")
        else:
            logger.info("AIHandler: AIService successfully initialized for Gynecology Medical AI.")

    def _is_question(self, message: str) -> bool:
        """Determine if the user message is a question."""
        if not message or not message.strip():
            return False
        
        message = message.lower().strip()
        
        # Check for question marks or patterns
        if message.endswith('?'):
            return True
        
        question_patterns = [
            r'^(what|how|where|when|why|who|which|can|could|would|do|does|is|are|will)\b',
            r'je\s', r'ninaweza', r'wapi', r'how much', r'how long', r'result',
            r'matokeo', r'kifaa', r'screening', r'kit'
        ]
        
        return any(re.search(pattern, message, re.IGNORECASE) for pattern in question_patterns)

    def handle_ai_chat_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict:
        """Handle ongoing AI chat state."""
        logger.info(f"AIHandler: Processing AI chat message for session {session_id}")
        return self._process_user_question(state, session_id, original_message)

    def handle_ai_menu_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict:
        """Handle AI chat selection state."""
        logger.info(f"AIHandler: Handling message '{message}' in AI menu state for session {session_id}. Original: '{original_message}'")
        if message == "ai_chat" or message == "start_ai_chat" or message == "initial_greeting":
            return self._handle_ai_chat_start(state, session_id, original_message)
        elif message == "back_to_main" or message == "menu":
            return self.handle_back_to_main(state, session_id)
        else:
            return self._handle_ai_chat_start(state, session_id, original_message)

    def handle_kit_request_collection_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict:
        """Handle collection of missing information for kit requests."""
        phone_number = state.get("phone_number", session_id)
        user_name = state.get("user_name", "Guest")
        kit_data = state.get("kit_request_data", {})
        location = kit_data.get("location")
        name = kit_data.get("name") or user_name

        logger.info(f"Handling kit request collection for session {session_id}: location={location}, name={name}")

        # Extract additional info from the message
        if not location:
            location = self.ai_service._extract_location(original_message)
        if not name or name == "Guest":
            name = self.ai_service._extract_name(original_message) or user_name

        # Check if all required info is now available
        if location and name and name != "Guest":
            try:
                response = (
                    f"Thank you, {name}! Your request for a cervical screening kit in {location} has been received.\n"
                    "A Community Health Worker will contact you soon to arrange delivery. Anything else I can help with?"
                )
                if self.ai_service._is_swahili(original_message):
                    response = (
                        f"Asante, {name}! Ombi lako la kifaa cha uchunguzi wa saratani ya shingo ya kizazi huko {location} limepokewa.\n"
                        "Mhudumu wa Afya ya Jamii atakupigia simu hivi karibuni kupanga utoaji. Je, kuna jambo lingine laweza kukusaidia?"
                    )
                
                # Reset state to ai_chat
                state["current_state"] = "ai_chat"
                state["kit_request_data"] = {}
                self.session_manager.update_session_state(session_id, state)
                
                return self.whatsapp_service.create_text_message(session_id, response)
            
            except Exception as e:
                logger.error(f"Error processing kit request for {phone_number}: {e}", exc_info=True)
                response = (
                    "Sorry, I couldn't process your kit request right now. Please try again or contact a health worker."
                )
                if self.ai_service._is_swahili(original_message):
                    response = (
                        "Samahani, sikuweza kushughulikia ombi lako la kifaa sasa hivi. Tafadhali jaribu tena au wasiliana na mhudumu wa afya."
                    )
                return self.whatsapp_service.create_text_message(session_id, response)
        
        # Still missing info
        missing = []
        if not location:
            missing.append("your area or county")
        if not name or name == "Guest":
            missing.append("your name")
        
        response = (
            f"Please provide {', and '.join(missing)} to complete your screening kit request."
        )
        if self.ai_service._is_swahili(original_message):
            response = (
                f"Tafadhali toa {', na '.join(missing)} ili kukamilisha ombi lako la kifaa cha uchunguzi."
            )
        
        state["kit_request_data"] = {"location": location, "name": name, "phone_number": phone_number}
        self.session_manager.update_session_state(session_id, state)
        return self.whatsapp_service.create_text_message(session_id, response)

    def _handle_ai_chat_start(self, state: Dict, session_id: str, user_message: str = None) -> Dict:
        """Handle AI chat start, setting state and processing user questions or greetings."""
        
        # Set up the chat state
        state["current_state"] = "ai_chat"
        state["current_handler"] = "ai_handler"
        
        # Initialize conversation history if not exists
        if "conversation_history" not in state:
            state["conversation_history"] = []
        
        user_name = state.get("user_name", "Guest")
        
        # Check if welcome message needs to be sent
        welcome_needed = not state.get("welcome_sent", False)
        
        # Determine if the user message is a question or a greeting
        is_question = self._is_question(user_message) if user_message else False
        has_user_message = user_message and user_message.strip() and user_message.lower() not in [
            "ai_chat", "start_ai_chat", "initial_greeting", "menu", "start", "hello", "hi", "hi there",
            "habari", "mambo"
        ]
        
        # Build the welcome message
        welcome_message = (
            f"Hi {user_name}! I'm AfyaBot, your gynecology medical assistant.\n\n"
            "I'm here to help with:\n"
            "- Cervical screening information\n"
            "- Finding Community Health Workers\n"
            "- Ordering screening kits\n"
            "- Questions about women's reproductive health\n\n"
            "What would you like to know or do today?"
        )
        is_swahili = self.ai_service._is_swahili(user_message) if user_message else False
        if is_swahili:
            welcome_message = (
                f"Habari {user_name}! Mimi ni AfyaBot, msaidizi wako wa afya ya uzazi.\n\n"
                "Niko hapa kukusaidia na:\n"
                "- Taarifa za uchunguzi wa saratani ya shingo ya kizazi\n"
                "- Kupata Wahudumu wa Afya ya Jamii\n"
                "- Kuomba vifaa vya uchunguzi\n"
                "- Maswali kuhusu afya ya uzazi ya wanawake\n\n"
                "Ungependa kujua nini au kufanya nini leo?"
            )
        
        # Update session state
        state["welcome_sent"] = True
        self.session_manager.update_session_state(session_id, state)
        
        # Handle cases based on message type
        if is_question and has_user_message:
            logger.info(f"Processing user question on chat start: {user_message[:50]}")
            return self._process_user_question(state, session_id, user_message)
        else:
            if welcome_needed:
                return self.whatsapp_service.create_text_message(session_id, welcome_message)
            return self.whatsapp_service.create_text_message(session_id, welcome_message)

    def _process_user_question(self, state: Dict, session_id: str, user_message: str) -> Dict:
        """Process user question without sending welcome message."""
        phone_number = state.get("phone_number", session_id)
        user_name = state.get("user_name", "Guest")
        
        logger.info(f"Processing question for {phone_number}: {user_message[:100] if user_message else 'None'}")
        
        if not self.ai_service.ai_enabled:
            error_message = (
                "Sorry, the AI chat feature is currently unavailable. Please contact a health provider directly."
            )
            if user_message and self.ai_service._is_swahili(user_message):
                error_message = (
                    "Samahani, kipengele cha gumzo la AI hakipatikani sasa hivi. Tafadhali wasiliana na mtoa huduma wa afya moja kwa moja."
                )
            return self.whatsapp_service.create_text_message(session_id, error_message)
        
        try:
            # Get conversation history
            conversation_history = state.get("conversation_history", [])
            
            logger.info(f"Calling AI service for message: {user_message[:100]}")
            
            # Generate AI response
            ai_response, needs_info, location_or_email, name = self.ai_service.generate_medical_response(
                user_message, 
                conversation_history,
                phone_number,
                user_name,
                session_id
            )
            
            logger.info(f"AI response generated: {ai_response[:100] if ai_response else 'None'}")
            
            # Handle info collection
            if needs_info:
                state["current_state"] = "kit_request_collection"
                state["kit_request_data"] = {
                    "location": location_or_email,
                    "name": name,
                    "phone_number": phone_number
                }
                self.session_manager.update_session_state(session_id, state)
                logger.info(f"Transitioning to kit_request_collection state for {phone_number}")
                return self.whatsapp_service.create_text_message(session_id, ai_response)
            
            # Update conversation history
            conversation_history.append({
                "user": user_message,
                "assistant": ai_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 exchanges
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            
            state["conversation_history"] = conversation_history
            self.session_manager.update_session_state(session_id, state)
            
            return self.whatsapp_service.create_text_message(session_id, ai_response)
        
        except Exception as e:
            logger.error(f"Error processing user question for session {session_id}: {e}", exc_info=True)
            error_message = (
                "I'm having trouble processing your request right now. "
                "Please contact a health provider directly."
            )
            if user_message and self.ai_service._is_swahili(user_message):
                error_message = (
                    "Nina shida kushughulikia ombi lako sasa hivi. "
                    "Tafadhali wasiliana na mtoa huduma wa afya moja kwa moja."
                )
            return self.whatsapp_service.create_text_message(session_id, error_message)