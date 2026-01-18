import logging
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

class AIService:
    """
    Enhanced AI-powered conversational service for BEDC WhatsApp Support Bot
    using LLM for intent detection and response generation with PostgreSQL.
    """

    def __init__(self, config, db_manager):
        """Initialize AI Service with LLM capabilities and database manager."""
        logger.info("=" * 60)
        logger.info("AIService initialization started")
        logger.info("=" * 60)
        
        self.db_manager = db_manager
        logger.info("✓ Database manager assigned")
        
        # Get OpenAI API key
        logger.info("Searching for OpenAI API key...")
        try:
            if isinstance(config, dict):
                self.openai_api_key = config.get("openai_api_key") or config.get("OPENAI_API_KEY")
                logger.info(f"Config is dict, API key found: {bool(self.openai_api_key)}")
            else:
                self.openai_api_key = getattr(config, 'OPENAI_API_KEY', os.getenv("OPENAI_API_KEY"))
                logger.info(f"Config is object, API key found: {bool(self.openai_api_key)}")
        except Exception as e:
            logger.warning(f"Error getting OpenAI API key from config: {e}")
            self.openai_api_key = None
        
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            logger.info(f"Checked environment variable, API key found: {bool(self.openai_api_key)}")
        
        # Initialize OpenAI client if available
        self.client = None
        self.ai_enabled = False
        
        logger.info(f"OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")
        logger.info(f"API Key present: {bool(self.openai_api_key)}")
        
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                logger.info("Initializing OpenAI client...")
                self.client = OpenAI(api_key=self.openai_api_key)
                self.ai_enabled = True
                logger.info("✓ AI Service initialized with LLM support")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                self.ai_enabled = False
        else:
            if not OPENAI_AVAILABLE:
                logger.warning("⚠ OpenAI library not available")
            if not self.openai_api_key:
                logger.warning("⚠ OpenAI API key not found")
            logger.warning("AI Service running in fallback mode (pattern matching only)")
            self.ai_enabled = False
        
        # Load FAQ knowledge base
        logger.info("Loading FAQ knowledge base...")
        self.faq_knowledge = self._load_faq_knowledge()
        logger.info(f"✓ FAQ knowledge base loaded ({len(self.faq_knowledge)} characters)")
        
        logger.info("=" * 60)
        logger.info(f"✅ AIService initialization completed (AI enabled: {self.ai_enabled})")
        logger.info("=" * 60)

    def _load_faq_knowledge(self) -> str:
        """Load FAQ knowledge base for the LLM."""
        return """FAQ KNOWLEDGE BASE:

IMPORTANT TERMINOLOGY:
- **Unmetered customers**: Customers WITHOUT a prepaid or digital meter. They receive estimated monthly bills based on NERC caps.
- **Prepaid meter**: A meter you purchase that allows you to pay for electricity in advance (buy units as needed).

1. **What is NERC capping?**
   NERC (Nigerian Electricity Regulatory Commission) capping is the maximum amount an unmetered customer can be charged monthly based on their feeder classification.

2. **Why is my bill higher than the NERC cap?**
   Bills above the NERC cap are billing errors. We apologize for this. Your account will be reviewed and adjusted within one billing cycle.

3. **How do I apply for a prepaid meter?**
   Visit https://imaap.beninelectric.com:55682/ and follow the MAP (Meter Asset Provider) enrollment process. You'll need your account number.

4. **What is MAP?**
   MAP stands for Meter Asset Provider - a scheme where you purchase your prepaid meter directly from approved vendors.

5. **Can I use one account for multiple meter applications?**
   No. Each meter location requires a separate postpaid account number. You cannot use one account for multiple meter applications.

6. **I'm a new customer without an account. How do I get a meter?**
   You must first visit our office at Ring Road, Benin City to create a postpaid account. Then you can apply for a meter through MAP.

7. **What documents do I need for a new account?**
   Bring valid ID, proof of address, and a utility bill (if available) to our office.

8. **How long does meter installation take?**
   After payment through MAP, installation is typically scheduled within 2-4 weeks.

9. **How do I report a power outage?**
   Provide your account number, phone number, and email. We'll log your report and our technical team will respond within 24-48 hours.

10. **What are BEDC's office hours?**
    Monday-Friday, 8:00 AM - 4:00 PM at Ring Road, Benin City.

11. **How much does a prepaid meter cost?**
    Meter costs vary by type (single-phase vs three-phase). Visit https://imaap.beninelectric.com:55682/ for current pricing.

12. **Can I pay my bill through WhatsApp?**
    Currently, bill payments are not available via WhatsApp. Please visit our office or use bank channels.

13. **What is my feeder?**
    Your feeder is shown on your bill. It determines your NERC cap amount. Contact us with your account number to confirm.

14. **How is the NERC cap calculated?**
    NERC caps are based on feeder classification and customer category. Unmetered customers are charged estimated consumption within these caps.

15. **Can I switch from postpaid to prepaid?**
    Yes! Enroll in MAP to get a prepaid meter. Your postpaid account will be closed once the meter is installed.

16. **What if I disagree with my bill?**
    Provide your account number. We'll review it against NERC caps and adjust if there's an error.

17. **How do I check my current bill?**
    Provide your account number and we'll check your billing status immediately.

18. **What areas does BEDC serve?**
    BEDC serves Edo, Delta, Ondo, and Ekpoma areas with multiple feeders across these regions.

19. **Can I get a meter if I have unpaid bills?**
    Outstanding bills should be cleared before MAP enrollment. Contact our office for payment arrangements.

20. **How do I update my contact information?**
    Visit our office with valid ID to update your phone number, email, or address on your account.
"""

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return f"""You are a helpful customer support representative for Benin Electricity Distribution Company (BEDC). You communicate naturally like a real person, not a bot.

CRITICAL RULES:
1. BE NATURAL - Sound like a real BEDC customer service rep, not a bot
2. CHECK CONTEXT - If user already provided information, DON'T ask again
3. ONE QUESTION AT A TIME - If you need info, ask for ONE thing only
4. NO REPETITION - Don't repeat what you already said
5. FAULT CONFIRMATION - Before logging a fault, ALWAYS confirm customer details

TONE GUIDELINES:
- **Apologize & Empathize** when: Billing errors, power outages, service failures
- **Be Direct & Helpful** when: Providing information, answering questions, giving instructions
- **Be Warm** for: Greetings, confirmations, simple requests
- **Be Polite** when asking: Use "Could you share..." not "What's your..."
- **Be Detailed** when explaining: Billing status, why something is correct/incorrect
- **Be Concise** for: Simple how-to questions, straightforward requests

RESPONSE FORMAT (JSON only):
{{
  "intent": "<Greeting | Billing | Metering | Fault | FaultConfirmation | FAQ>",
  "reply": "<Natural, conversational response>",
  "required_data": ["<what's still missing>"]
}}

INTENT HANDLING:

**Greeting**: 
- Warm and natural: "Hi! I'm here to help with your BEDC account. What can I assist you with today?"

**Billing**: 
- If no account → "Could you share your account number so I can look into your billing?"
- If bill ABOVE CAP → "I sincerely apologize for this error. Your bill of ₦X,XXX exceeds the ₦Y,YYY NERC cap by ₦Z,ZZZ. We'll adjust it within one billing cycle."
- If bill WITHIN CAP (unmetered) → "Your bill of ₦X,XXX is within the ₦Y,YYY NERC cap for [Feeder Name] feeder. Since you're an unmetered customer (no prepaid meter), your billing follows the approved methodology. For more accurate billing, you can apply for a prepaid meter at https://imaap.beninelectric.com:55682/"
- If bill ABOVE CAP (unmetered) → "I sincerely apologize for this error. Your bill of ₦X,XXX exceeds the ₦Y,YYY NERC cap by ₦Z,ZZZ. We'll adjust it within one billing cycle. Since you're unmetered, you can get a prepaid meter for more accurate billing at https://imaap.beninelectric.com:55682/"

**Fault**:
- ALWAYS apologize for power outages
- "I'm sorry about the outage. Let me help you log a fault report. Could you share your account number?" (if needed)
- Once you have account + email → Use "FaultConfirmation" intent to confirm before logging

**FaultConfirmation**:
- Before logging, confirm: "Just to confirm - is this your account?\n\nAccount: 123456\nEmail: user@email.com\n\nReply 'Yes' to confirm or 'No' to update."
- After "Yes" confirmation → Log the fault with success message
- After "No" → Ask what needs updating

**Metering**:
- Be helpful and direct: "You can apply for a prepaid meter at https://imaap.beninelectric.com:55682/"
- If they ask what unmetered means → "Being unmetered means you don't have a prepaid meter yet. You receive estimated monthly bills based on NERC caps. Getting a prepaid meter lets you pay for only what you use."
- If they need steps → Give 2-3 simple steps
- If new customer → "You'll need to visit our office at Ring Road, Benin City to create an account first."

**FAQ**: 
- Answer directly from knowledge base in 1-3 sentences
- Be helpful, not overly formal

FAQ KNOWLEDGE:
{self.faq_knowledge}

EXAMPLES:

✅ GOOD - Greeting (Natural):
"Hi! I'm here to help with your BEDC account. What can I assist you with today?"

✅ GOOD - Billing Within Cap (Detailed with clarification):
"Your bill of ₦14,500 is within the ₦15,000 NERC cap for Uselu feeder. Since you're an unmetered customer (no prepaid meter), your billing follows the approved methodology. For more accurate billing, consider applying for a prepaid meter at https://imaap.beninelectric.com:55682/"

✅ GOOD - Fault Confirmation:
"Just to confirm - is this your account?

Account: 102345
Email: john@email.com

Reply 'Yes' to confirm or 'No' to update."

✅ GOOD - After Confirmation:
"Fault report logged successfully!

Reference: FR-102345-20260117
Email: john@email.com

Our technical team will contact you within 24-48 hours. Thanks for your patience."

❌ BAD - Robotic greeting:
"Hello! I'm BEDC Support Bot. How can I help you?"

❌ BAD - Skipping confirmation:
*Logs fault immediately without asking for confirmation*

BALANCE: Be human, caring, and natural. Avoid sounding like a bot."""

    def call_llm(self, user_message: str, conversation_state: Dict = None,
                 customer_data: Dict = None, billing_result: Dict = None,
                 conversation_history: List[Dict] = None) -> Dict:
        """
        Call the LLM to process user message and return structured response.
        
        Returns:
            Dict with keys: intent, reply, required_data
        """
        if not self.ai_enabled or not self.client:
            # Fallback to pattern matching
            return self._fallback_response(user_message, customer_data, billing_result)
        
        try:
            # Prepare context
            context_parts = []
            
            # Add conversation history for context
            if conversation_history and len(conversation_history) > 0:
                recent_history = conversation_history[-5:]  # Last 5 exchanges
                history_text = "PREVIOUS CONVERSATION:\n"
                for exchange in recent_history:
                    history_text += f"User: {exchange.get('user', '')}\n"
                    history_text += f"You: {exchange.get('assistant', '')}\n"
                context_parts.append(history_text)
            
            if conversation_state:
                state_info = f"SESSION DATA:\n"
                if conversation_state.get("saved_account_number"):
                    state_info += f"- Account: {conversation_state['saved_account_number']}\n"
                if conversation_state.get("email_from_database"):
                    state_info += f"- Email: ALREADY IN DATABASE (don't ask for it)\n"
                elif conversation_state.get("has_email"):
                    state_info += f"- Email: Already provided\n"
                if conversation_state.get("phone_number"):
                    state_info += f"- Phone: {conversation_state['phone_number']}\n"
                if conversation_state.get("pending_fault_confirmation"):
                    state_info += f"- PENDING FAULT CONFIRMATION: Waiting for user to confirm details\n"
                context_parts.append(state_info)
            
            if customer_data:
                context_parts.append(f"CUSTOMER DATA: {json.dumps(customer_data, default=str)}")
            
            if billing_result:
                context_parts.append(f"BILLING STATUS: {json.dumps(billing_result, default=str)}")
            
            context = "\n".join(context_parts) if context_parts else "No additional context."
            
            # Construct user prompt
            user_prompt = f"""Customer message: "{user_message}"

{context}

Remember: Be NATURAL (not robotic). Check context before asking for info. For faults, ALWAYS confirm details before logging. Return JSON only."""
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Validate required fields
            if "intent" not in result or "reply" not in result:
                raise ValueError("LLM response missing required fields")
            
            if "required_data" not in result:
                result["required_data"] = []
            
            logger.info(f"LLM detected intent: {result['intent']}")
            return result
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            return self._fallback_response(user_message, customer_data, billing_result)

    def _fallback_response(self, user_message: str, customer_data: Dict = None,
                           billing_result: Dict = None) -> Dict:
        """Fallback pattern-matching based response when LLM is unavailable."""
        message_lower = user_message.lower().strip()
        
        # Greeting - warm and natural
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return {
                "intent": "Greeting",
                "reply": "Hi! I'm here to help with your BEDC account. What can I assist you with today?",
                "required_data": []
            }
        
        # Billing - empathetic for errors, detailed for information
        if any(word in message_lower for word in ['bill', 'billing', 'charge', 'overcharge', 'nerc', 'cap']):
            if billing_result:
                customer = billing_result.get('customer_data', {})
                feeder = customer.get('feeder', 'your')
                is_metered = customer.get('metered', False)
                
                if billing_result['status'] == 'within_cap':
                    reply = f"Your bill of ₦{billing_result['bill_amount']:,} is within the ₦{billing_result['nerc_cap']:,} NERC cap for {feeder} feeder. "
                    if not is_metered:
                        reply += "Since you're an unmetered customer (no prepaid meter), your billing follows the approved methodology. "
                        reply += "For more accurate billing, consider applying for a prepaid meter at https://imaap.beninelectric.com:55682/"
                else:
                    # Apologize for billing errors with details
                    reply = f"I sincerely apologize for this error. Your bill of ₦{billing_result['bill_amount']:,} exceeds the ₦{billing_result['nerc_cap']:,} NERC cap by ₦{billing_result['difference']:,}. "
                    reply += "We'll adjust it within one billing cycle."
                    if not is_metered:
                        reply += " Since you're unmetered, you can get a prepaid meter for more accurate billing at https://imaap.beninelectric.com:55682/"
                return {"intent": "Billing", "reply": reply, "required_data": []}
            else:
                return {
                    "intent": "Billing",
                    "reply": "Could you share your account number so I can look into your billing?",
                    "required_data": ["account_number"]
                }
        
        # Metering - direct and helpful
        if any(word in message_lower for word in ['meter', 'prepaid', 'map', 'apply', 'unmetered']):
            return {
                "intent": "Metering",
                "reply": "You can apply for a prepaid meter at https://imaap.beninelectric.com:55682/",
                "required_data": []
            }
        
        # Fault - always apologize for service issues
        if any(word in message_lower for word in ['fault', 'outage', 'no power', 'blackout', 'no light']):
            return {
                "intent": "Fault",
                "reply": "I'm sorry about the outage. Let me help you log a fault report. Could you share your account number?",
                "required_data": ["account_number", "email"]
            }
        
        # Default - helpful
        return {
            "intent": "FAQ",
            "reply": "I can help with billing, meters, or fault reports. What do you need assistance with?",
            "required_data": []
        }

    def extract_account_number(self, message: str) -> Optional[str]:
        """Extract account number from message."""
        if not message:
            return None
        match = re.search(r'\b(10\d{4})\b', message)
        return match.group(1) if match else None

    def extract_email(self, message: str) -> Optional[str]:
        """Extract email from message."""
        if not message:
            return None
        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message)
        return match.group(0) if match else None

    def generate_response(self, user_message: str, conversation_history: List[Dict] = None,
                         phone_number: str = None, user_name: str = None,
                         session_state: Dict = None) -> Tuple[str, str, Dict]:
        """
        Generate AI response using LLM with proper flow handling.

        Returns:
            Tuple[str, str, Dict]: (response, intent, updated_state)
        """
        if not user_message or not isinstance(user_message, str):
            return ("I'm sorry, I didn't receive a valid message. How can I help you?", "unknown", {})
        
        if session_state is None:
            session_state = {}
        
        # Check if we're waiting for fault confirmation
        pending_confirmation = session_state.get("pending_fault_confirmation", False)
        
        # Handle confirmation responses
        if pending_confirmation:
            confirmation_lower = user_message.lower().strip()
            if confirmation_lower in ['yes', 'y', 'confirm', 'correct', 'ok', 'okay']:
                # Proceed with logging the fault
                fault_data = session_state.get("fault_data", {})
                
                success = self.db_manager.save_fault_report(
                    fault_data["phone_number"],
                    fault_data["account_number"],
                    fault_data["email"],
                    fault_data.get("fault_description", "Power outage reported")
                )
                
                if success:
                    reply = "Fault report logged successfully!\n\n"
                    reply += f"Reference: FR-{fault_data['account_number']}-{datetime.now().strftime('%Y%m%d')}\n"
                    reply += f"Email: {fault_data['email']}\n\n"
                    reply += "Our technical team will contact you within 24-48 hours. Thanks for your patience."
                    
                    return (reply, "Fault", {
                        "fault_data": {},
                        "pending_fault_confirmation": False
                    })
                else:
                    return ("I apologize, but there was an error logging your fault report. Please try again.", "Fault", {
                        "pending_fault_confirmation": False
                    })
            
            elif confirmation_lower in ['no', 'n', 'wrong', 'incorrect', 'update']:
                reply = "No problem. What would you like to update? Please provide the correct account number or email."
                return (reply, "Fault", {
                    "pending_fault_confirmation": False,
                    "fault_data": {}
                })
        
        # Check if account number already exists in session
        saved_account_number = session_state.get("account_number")
        
        # Extract account number and email if present in current message
        account_number = self.extract_account_number(user_message)
        email = self.extract_email(user_message)
        
        # Use saved account number if no new one found
        if not account_number and saved_account_number:
            account_number = saved_account_number
            logger.info(f"Using saved account number: {account_number}")
        
        # Get customer data if account number is available
        customer_data = None
        billing_result = None
        customer_email = None
        
        if account_number:
            customer_data = self.db_manager.get_customer_by_account(account_number)
            if customer_data:
                billing_result = self.db_manager.check_billing_status(account_number)
                customer_email = customer_data.get("email")
                logger.info(f"Found customer email in database: {customer_email}")
        
        # Use database email if user hasn't provided one
        if not email and customer_email:
            email = customer_email
            logger.info(f"Using email from customer database: {email}")
        
        # Prepare conversation state for LLM
        conversation_state = {
            "phone_number": phone_number,
            "user_name": user_name,
            "has_account_number": bool(account_number),
            "has_email": bool(email),
            "email_from_database": bool(customer_email),
            "session_data": session_state,
            "saved_account_number": saved_account_number,
            "pending_fault_confirmation": pending_confirmation
        }
        
        if account_number:
            conversation_state["account_number"] = account_number
        
        # Call LLM to get response
        llm_response = self.call_llm(
            user_message,
            conversation_state,
            customer_data,
            billing_result,
            conversation_history
        )
        
        intent = llm_response.get("intent", "unknown")
        reply = llm_response.get("reply", "I'm here to help. What can I assist you with?")
        required_data = llm_response.get("required_data", [])
        
        # Handle specific flows based on intent
        state_update = {}
        
        # Save account number to session if found
        if account_number:
            state_update["account_number"] = account_number
            logger.info(f"Saving account number {account_number} to session")
        
        if intent == "Fault":
            # Handle fault reporting flow
            fault_data = session_state.get("fault_data", {})
            
            if account_number and account_number not in fault_data:
                fault_data["account_number"] = account_number
            if email and "email" not in fault_data:
                fault_data["email"] = email
            if phone_number and "phone_number" not in fault_data:
                fault_data["phone_number"] = phone_number
            if not fault_data.get("fault_description"):
                fault_data["fault_description"] = user_message
            
            # Check if we have all required data
            if fault_data.get("account_number") and fault_data.get("email") and fault_data.get("phone_number"):
                # Ask for confirmation before logging
                confirmation_message = f"Just to confirm - is this your account?\n\n"
                confirmation_message += f"Account: {fault_data['account_number']}\n"
                confirmation_message += f"Email: {fault_data['email']}\n\n"
                confirmation_message += "Reply 'Yes' to confirm or 'No' to update."
                
                state_update["fault_data"] = fault_data
                state_update["pending_fault_confirmation"] = True
                
                return (confirmation_message, "FaultConfirmation", state_update)
            else:
                state_update["fault_data"] = fault_data
        
        elif intent == "Billing" and account_number:
            # Save billing inquiry
            state_update["account_number"] = account_number
            if billing_result:
                state_update["billing_checked"] = True
        
        logger.info(f"Generated response for intent '{intent}': {reply[:100]}")
        
        return (reply, intent, state_update)