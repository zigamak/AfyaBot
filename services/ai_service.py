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
    using LLM for intent detection and response generation.
    """

    def __init__(self, config, data_manager):
        """Initialize AI Service with LLM capabilities."""
        self.data_manager = data_manager
        
        # Get OpenAI API key
        try:
            if isinstance(config, dict):
                self.openai_api_key = config.get("openai_api_key")
            else:
                self.openai_api_key = getattr(config, 'OPENAI_API_KEY', os.getenv("OPENAI_API_KEY"))
        except:
            self.openai_api_key = None
        
        # Initialize OpenAI client if available
        self.client = None
        self.ai_enabled = False
        
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
                self.ai_enabled = True
                logger.info("AI Service initialized with LLM support")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.ai_enabled = False
        else:
            logger.warning("AI Service running in fallback mode (pattern matching only)")
            self.ai_enabled = False
        
        # Load FAQ knowledge base
        self.faq_knowledge = self._load_faq_knowledge()

    def _load_faq_knowledge(self) -> str:
        """Load FAQ knowledge base for the LLM."""
        return """FAQ KNOWLEDGE BASE:

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
        return f"""You are an AI customer support assistant for Benin Electricity Distribution Company (BEDC).

CRITICAL RULES:
1. BE CONCISE but APPROPRIATE - Match tone to situation
2. CHECK CONTEXT - If user already provided information, DON'T ask again
3. ONE QUESTION AT A TIME - If you need info, ask for ONE thing only
4. NO REPETITION - Don't repeat what you already said
5. CONFIRM ACCOUNT NUMBER - Always verify account number exists in database before logging faults
6. UNDERSTAND NATURAL LANGUAGE - "my electricity went off", "no light", "power outage" all mean FAULT

TONE GUIDELINES:
- **Apologize & Empathize** when: Billing errors, power outages, service failures
- **Be Direct & Helpful** when: Providing information, answering questions, giving instructions
- **Be Warm** for: Greetings, confirmations, simple requests
- **Be Polite** when asking: Use "Please provide..." not "What's your..."
- **Be Detailed** when explaining: Billing status, why something is correct/incorrect
- **Be Concise** for: Simple how-to questions, straightforward requests

RESPONSE FORMAT (JSON only):
{{
  "intent": "<Greeting | Billing | Metering | Fault | FAQ>",
  "reply": "<Appropriate tone, concise response>",
  "required_data": ["<what's still missing>"],
  "account_verified": <true/false>
}}

INTENT HANDLING:

**Greeting**: 
- Warm and brief: "Hello! How can I help you today?"

**Billing**: 
- If no account → "Please provide your account number so I can review your billing."
- If bill ABOVE CAP → "I apologize for this error. Your bill of ₦X,XXX exceeds the ₦Y,YYY NERC cap by ₦Z,ZZZ. We'll adjust it within one billing cycle."
- If bill WITHIN CAP → "Your bill of ₦X,XXX is within the ₦Y,YYY NERC cap for [Feeder Name] feeder. Your billing follows the approved methodology for unmetered customers. For more accurate billing, consider applying for a prepaid meter at https://imaap.beninelectric.com:55682/"

**Fault** (MOST IMPORTANT):
- Recognize: "my electricity went off", "no light", "no power", "blackout", "outage", "power failure"
- ALWAYS apologize for power outages first
- If NO account number → "I sincerely apologize for the power outage. Please provide your account number so I can log this fault."
- If account NOT in database → "I couldn't find that account number. Please verify and provide the correct account number."
- If account FOUND but NO email → "I'll log this for account {number}. Please provide your email address."
- If account FOUND with email → Show confirmation: "I'll log this fault for account {number} ({name}, {feeder} feeder, email: {email}). Reply 'yes' to confirm or 'no' to correct."
- After confirmation → "Fault logged (Ref: FR-XXXXX). Our team will contact you within 24-48 hours."

**Metering**:
- Be helpful and direct: "You can apply for a prepaid meter at https://imaap.bedinelectric.com:55682/"
- If they need steps → Give 2-3 simple steps
- If new customer → "You'll need to visit our office at Ring Road, Benin City to create an account first."

**FAQ**: 
- Answer directly from knowledge base in 1-3 sentences
- Be helpful, not overly formal

FAQ KNOWLEDGE:
{self.faq_knowledge}

EXAMPLES OF FAULT RECOGNITION:

User: "My electricity went off"
→ Intent: Fault (NOT FAQ)
→ Reply: "I sincerely apologize for the power outage. Please provide your account number so I can log this fault."

User: "No light since yesterday"
→ Intent: Fault (NOT FAQ)
→ Reply: "I apologize for the ongoing power outage. What's your account number?"

User: "Power failure in my area"
→ Intent: Fault (NOT FAQ)
→ Reply: "I'm sorry about the power outage. Please share your account number to log this fault."

✅ GOOD - Fault Confirmation:
"I'll log this fault for account 101234 (John Doe, Uselu feeder, email: john@email.com). Reply 'yes' to confirm or 'no' to correct."

✅ GOOD - Account Not Found:
"I couldn't find that account number in our system. Please verify and provide the correct account number."

✅ GOOD - Need Email:
"I'll log this for account 101234. Please provide your email address."

BALANCE: Be human and caring when users have problems. Be efficient and helpful when they need information."""

    def call_llm(self, user_message: str, conversation_state: Dict = None,
                 customer_data: Dict = None, billing_result: Dict = None,
                 conversation_history: List[Dict] = None) -> Dict:
        """
        Call the LLM to process user message and return structured response.
        
        Returns:
            Dict with keys: intent, reply, required_data, account_verified
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
                if conversation_state.get("awaiting_confirmation"):
                    state_info += f"- Status: AWAITING USER CONFIRMATION FOR FAULT LOGGING\n"
                context_parts.append(state_info)
            
            if customer_data:
                context_parts.append(f"CUSTOMER DATA (VERIFIED IN DATABASE): {json.dumps(customer_data)}")
            
            if billing_result:
                context_parts.append(f"BILLING STATUS: {json.dumps(billing_result)}")
            
            context = "\n".join(context_parts) if context_parts else "No additional context."
            
            # Construct user prompt
            user_prompt = f"""Customer message: "{user_message}"

{context}

Remember: Recognize fault reports in natural language. ALWAYS verify account before logging. Ask for email if not in database. Return JSON only."""
            
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
            
            if "account_verified" not in result:
                result["account_verified"] = bool(customer_data)
            
            logger.info(f"LLM detected intent: {result['intent']}")
            return result
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            return self._fallback_response(user_message, customer_data, billing_result)

    def _fallback_response(self, user_message: str, customer_data: Dict = None,
                           billing_result: Dict = None) -> Dict:
        """Fallback pattern-matching based response when LLM is unavailable."""
        message_lower = user_message.lower().strip()
        
        # Greeting - warm and welcoming
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return {
                "intent": "Greeting",
                "reply": "Hello! How can I help you today?",
                "required_data": [],
                "account_verified": False
            }
        
        # Fault - recognize natural language (MOST COMPREHENSIVE)
        fault_keywords = [
            'fault', 'outage', 'no power', 'blackout', 'no light', 'power off',
            'electricity off', 'electricity went off', 'power went off', 'went off',
            'power failure', 'power problem', 'no electricity', 'light off',
            'light went off', 'nepa', 'phcn'
        ]
        
        if any(keyword in message_lower for keyword in fault_keywords):
            if customer_data:
                # Account verified, ask for confirmation
                customer_name = customer_data.get("customer_name", "")
                feeder = customer_data.get("feeder", "")
                email = customer_data.get("email", "")
                account_number = customer_data.get("account_number", "")
                
                if email:
                    reply = f"I sincerely apologize for the power outage.\n\n"
                    reply += f"I'll log this fault for account {account_number}"
                    if customer_name:
                        reply += f" ({customer_name}"
                        if feeder:
                            reply += f", {feeder} feeder"
                        reply += f", email: {email})"
                    reply += ".\n\nReply 'yes' to confirm or 'no' to correct."
                    return {
                        "intent": "Fault",
                        "reply": reply,
                        "required_data": [],
                        "account_verified": True
                    }
                else:
                    reply = f"I sincerely apologize for the power outage.\n\n"
                    reply += f"I'll log this for account {account_number}"
                    if customer_name:
                        reply += f" ({customer_name})"
                    reply += ". Please provide your email address."
                    return {
                        "intent": "Fault",
                        "reply": reply,
                        "required_data": ["email"],
                        "account_verified": True
                    }
            else:
                return {
                    "intent": "Fault",
                    "reply": "I sincerely apologize for the power outage. Please provide your account number so I can log this fault.",
                    "required_data": ["account_number"],
                    "account_verified": False
                }
        
        # Billing - empathetic for errors, detailed for information
        if any(word in message_lower for word in ['bill', 'billing', 'charge', 'overcharge', 'nerc', 'cap']):
            if billing_result:
                customer = billing_result.get('customer_data', {})
                feeder = customer.get('feeder', 'your')
                
                if billing_result['status'] == 'within_cap':
                    reply = f"Your bill of ₦{billing_result['bill_amount']:,} is within the ₦{billing_result['nerc_cap']:,} NERC cap for {feeder} feeder. "
                    reply += "Your billing follows the approved methodology for unmetered customers. "
                    reply += "For more accurate billing, consider applying for a prepaid meter at https://imaap.beninelectric.com:55682/"
                else:
                    reply = f"I sincerely apologize for this error. Your bill of ₦{billing_result['bill_amount']:,} exceeds the ₦{billing_result['nerc_cap']:,} NERC cap by ₦{billing_result['difference']:,}. "
                    reply += "We'll adjust it within one billing cycle."
                return {"intent": "Billing", "reply": reply, "required_data": [], "account_verified": True}
            else:
                return {
                    "intent": "Billing",
                    "reply": "Please provide your account number so I can review your billing.",
                    "required_data": ["account_number"],
                    "account_verified": False
                }
        
        # Metering - direct and helpful
        if any(word in message_lower for word in ['meter', 'prepaid', 'map', 'apply']):
            return {
                "intent": "Metering",
                "reply": "You can apply for a prepaid meter at https://imaap.beninelectric.com:55682/",
                "required_data": [],
                "account_verified": False
            }
        
        # FAQ - answer questions
        if any(word in message_lower for word in ['what', 'how', 'when', 'where', 'why', '?']):
            return {
                "intent": "FAQ",
                "reply": "I can help with billing issues, meter applications, and fault reports. What specifically would you like to know?",
                "required_data": [],
                "account_verified": False
            }
        
        # Default - helpful but brief
        return {
            "intent": "FAQ",
            "reply": "How can I assist you? I can help with billing, meters, or fault reports.",
            "required_data": [],
            "account_verified": False
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
        
        # Check if awaiting confirmation
        awaiting_confirmation = session_state.get("awaiting_fault_confirmation", False)
        
        logger.info(f"generate_response called. awaiting_confirmation={awaiting_confirmation}, message='{user_message}'")
        
        # Handle confirmation responses
        if awaiting_confirmation:
            message_lower = user_message.lower().strip()
            if message_lower in ['yes', 'y', 'yeah', 'correct', 'ok', 'okay', 'confirm']:
                # User confirmed - proceed with logging
                fault_data = session_state.get("fault_data", {})
                
                logger.info(f"User confirmed. Logging fault: {fault_data}")
                
                success = self.data_manager.save_fault_report(
                    fault_data.get("phone_number"),
                    fault_data.get("account_number"),
                    fault_data.get("email"),
                    fault_data.get("fault_description", "Power outage reported")
                )
                
                if success:
                    reply = "✅ Fault report logged successfully!\n\n"
                    reply += f"📋 Reference: FR-{fault_data['account_number']}-{datetime.now().strftime('%Y%m%d')}\n"
                    reply += f"📧 Email: {fault_data['email']}\n\n"
                    reply += "Our technical team will contact you within 24-48 hours. Thank you for your patience."
                    state_update = {
                        "fault_data": {},
                        "awaiting_fault_confirmation": False
                    }
                    return (reply, "Fault", state_update)
                else:
                    reply = "I apologize, but there was an error logging your fault report. Please try again or contact our office at Ring Road, Benin City."
                    state_update = {"awaiting_fault_confirmation": False, "fault_data": {}}
                    return (reply, "Fault", state_update)
            
            elif message_lower in ['no', 'n', 'nope', 'incorrect', 'wrong']:
                # User declined - ask for correct account
                reply = "I apologize for the confusion. Please provide the correct account number."
                state_update = {
                    "fault_data": {},
                    "awaiting_fault_confirmation": False
                }
                return (reply, "Fault", state_update)
        
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
        account_verified = False
        
        if account_number:
            customer_data = self.data_manager.get_customer_by_account(account_number)
            if customer_data:
                account_verified = True
                billing_result = self.data_manager.check_billing_status(account_number)
                customer_email = customer_data.get("email")
                logger.info(f"✅ Account {account_number} verified. Customer: {customer_data.get('customer_name')}, Email: {customer_email}")
            else:
                logger.warning(f"❌ Account {account_number} NOT found in database")
        
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
            "awaiting_confirmation": awaiting_confirmation
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
        reply = llm_response.get("reply", "I'm here to help. Please tell me more about your concern.")
        required_data = llm_response.get("required_data", [])
        
        logger.info(f"LLM Intent: {intent}, Account Verified: {account_verified}")
        
        # Handle specific flows based on intent
        state_update = {}
        
        # Save account number to session if found and verified
        if account_number and account_verified:
            state_update["account_number"] = account_number
            logger.info(f"✅ Saving verified account number {account_number} to session")
        
        if intent == "Fault":
            # Handle fault reporting flow with confirmation
            fault_data = session_state.get("fault_data", {})
            
            # If account not verified, ask for valid account
            if account_number and not account_verified:
                reply = "I couldn't find that account number in our system. Please verify and provide the correct account number."
                state_update["fault_data"] = {}
                return (reply, intent, state_update)
            
            # Store fault data
            if account_number and account_verified:
                fault_data["account_number"] = account_number
            if email:
                fault_data["email"] = email
            if phone_number:
                fault_data["phone_number"] = phone_number
            if not fault_data.get("fault_description"):
                fault_data["fault_description"] = user_message
            
            logger.info(f"Fault data collected: {fault_data}")
            
            # Check if we have all required data and account is verified
            has_account = bool(fault_data.get("account_number"))
            has_email = bool(fault_data.get("email"))
            has_phone = bool(fault_data.get("phone_number"))
            
            logger.info(f"Fault requirements: account={has_account}, email={has_email}, phone={has_phone}, verified={account_verified}")
            
            if has_account and has_email and has_phone and account_verified:
                # Ask for confirmation before logging
                customer_name = customer_data.get("customer_name", "")
                feeder = customer_data.get("feeder", "")
                
                reply = f"I'll log this fault for account {fault_data['account_number']}"
                if customer_name:
                    reply += f" ({customer_name}"
                    if feeder:
                        reply += f", {feeder} feeder"
                    reply += f", email: {fault_data['email']})"
                else:
                    reply += f" (email: {fault_data['email']})"
                reply += ".\n\nReply 'yes' to confirm or 'no' to correct."
                
                state_update["fault_data"] = fault_data
                state_update["awaiting_fault_confirmation"] = True
                
                logger.info("✅ All fault data collected. Awaiting user confirmation.")
            else:
                state_update["fault_data"] = fault_data
                logger.info(f"⏳ Waiting for more fault data: {required_data}")
        
        elif intent == "Billing" and account_number:
            # Save billing inquiry
            state_update["account_number"] = account_number
            if billing_result:
                state_update["billing_checked"] = True
        
        logger.info(f"Generated response for intent '{intent}': {reply[:100]}")
        
        return (reply, intent, state_update)