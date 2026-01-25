import logging
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Azure OpenAI
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

class AIService:
    """
    Enhanced AI-powered conversational service for BEDC WhatsApp Support Bot
    
    Features:
    - Account validation
    - Email masking for privacy
    - Billing confirmation modal before showing data
    - Fault confirmation before logging
    - Smart confirmation detection
    """

    def __init__(self, config, data_manager):
        """Initialize AI Service with Azure OpenAI capabilities."""
        self.data_manager = data_manager
        
        # Get Azure OpenAI configuration
        try:
            if isinstance(config, dict):
                self.azure_api_key = config.get("AZURE_API_KEY")
                self.azure_endpoint = config.get("AZURE_ENDPOINT")
                self.azure_api_version = config.get("AZURE_API_VERSION", "2024-02-15-preview")
                self.azure_deployment = config.get("AZURE_DEPLOYMENT_NAME")
            else:
                # Config is an object, so use getattr
                self.azure_api_key = getattr(config, 'AZURE_API_KEY', None)
                self.azure_endpoint = getattr(config, 'AZURE_ENDPOINT', None)
                self.azure_api_version = getattr(config, 'AZURE_API_VERSION', "2024-02-15-preview")
                self.azure_deployment = getattr(config, 'AZURE_DEPLOYMENT_NAME', None)
        except Exception as e:
            logger.error(f"Error loading Azure OpenAI config: {e}")
            self.azure_api_key = None
            self.azure_endpoint = None
            self.azure_api_version = None
            self.azure_deployment = None
        
        # Initialize Azure OpenAI client if available
        self.client = None
        self.ai_enabled = False
        
        if OPENAI_AVAILABLE and self.azure_api_key and self.azure_endpoint and self.azure_deployment:
            try:
                self.client = AzureOpenAI(
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                    azure_endpoint=self.azure_endpoint
                )
                self.ai_enabled = True
                logger.info(f"AI Service initialized with Azure OpenAI support (Deployment: {self.azure_deployment})")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                self.ai_enabled = False
        else:
            missing = []
            if not OPENAI_AVAILABLE:
                missing.append("OpenAI library")
            if not self.azure_api_key:
                missing.append("AZURE_API_KEY")
            if not self.azure_endpoint:
                missing.append("AZURE_ENDPOINT")
            if not self.azure_deployment:
                missing.append("AZURE_DEPLOYMENT_NAME")
            
            logger.warning(f"AI Service running in fallback mode. Missing: {', '.join(missing)}")
            self.ai_enabled = False
        
        # Load FAQ knowledge base
        self.faq_knowledge = self._load_faq_knowledge()

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
"""

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return f"""You are a helpful customer support representative for Benin Electricity Distribution Company (BEDC). You communicate naturally like a real person, not a bot.

CRITICAL RULES:
1. BE NATURAL - Sound like a real BEDC customer service rep, not a bot
2. CHECK CONTEXT - If user already provided information, DON'T ask again
3. ONE QUESTION AT A TIME - If you need info, ask for ONE thing only
4. NO REPETITION - Don't repeat what you already said
5. CONFIRMATION BEFORE DATA - Before showing billing/account data, ALWAYS confirm account details first
6. VALIDATE ACCOUNT - If account doesn't exist, inform user clearly and helpfully

TONE GUIDELINES:
- **Apologize & Empathize** when: Billing errors, power outages, service failures
- **Be Direct & Helpful** when: Providing information, answering questions, giving instructions
- **Be Warm** for: Greetings, confirmations, simple requests
- **Be Polite** when asking: Use "Could you share..." not "What's your..."
- **Be Detailed** when explaining: Billing status, why something is correct/incorrect, next steps
- **Be Concise** for: Simple how-to questions, straightforward requests

RESPONSE FORMAT (JSON only):
{{
  "intent": "<Greeting | Billing | BillingConfirmation | Metering | Fault | FaultConfirmation | FAQ | AccountNotFound>",
  "reply": "<Natural, conversational response>",
  "required_data": ["<what's still missing>"]
}}

INTENT HANDLING:

**Greeting**: 
- Warm and natural: "Hi! I'm here to help with your BEDC account. What can I assist you with today?"
- If they greet AND ask about account → Ask for account number politely

**AccountNotFound**:
- ONLY show when user provides an account number that doesn't exist
- "I couldn't find an account associated with the number you provided. Please check and confirm:\n\n• Is the account number correct? (It should be 6 digits starting with '10')\n• If you don't have an account yet, visit our office at Ring Road, Benin City (Monday-Friday, 8AM-4PM) to create one. Bring valid ID and proof of address.\n\nYou can also provide a different account number if there was a typo."

**Billing**: 
- If account not found → STOP and return AccountNotFound intent instead
- If no account provided yet → "Could you share your account number so I can look into your billing?"
- If account provided and exists AND billing data available AND user has NOT been asked to confirm yet → Use "BillingConfirmation" intent to confirm details first (DON'T show billing data yet)
- IMPORTANT: If you see "PENDING BILLING CONFIRMATION" in session data, it means confirmation was ALREADY shown. DO NOT use BillingConfirmation intent again.

**BillingConfirmation**:
- Before showing billing data, confirm account details with masked email
- "Let me check your account. Is this correct?\n\nAccount: 123456\nEmail: ma****@gm***.com\n\nReply 'Yes' to proceed or 'No' to update."
- NOTE: Email should be masked for privacy
- After "Yes" confirmation → NOW show the billing data with full details
- After "No" → Ask what needs updating

**Fault**:
- If account not found → STOP and return AccountNotFound intent instead
- ALWAYS apologize for power outages first
- "I'm sorry about the outage. Let me help you log a fault report. Could you share your account number?" (if needed)
- Once you have VALID account + email → Use "FaultConfirmation" intent to confirm before logging

**FaultConfirmation**:
- Before logging fault, confirm all details: "Just to confirm - is this your account?\n\nAccount: 123456\nEmail: ma****@gm***.com\n\nReply 'Yes' to confirm or 'No' to update."
- NOTE: Email should be masked for privacy
- After "Yes" confirmation → Log the fault with success message
- After "No" → Ask what needs updating

**Metering**:
- Be helpful and direct: "You can apply for a prepaid meter at https://imaap.bedinelectric.com:55682/ - you'll need your account number."

**FAQ**: 
- Answer directly from knowledge base in 1-3 sentences

FAQ KNOWLEDGE:
{self.faq_knowledge}

BALANCE: Be human, caring, and natural. Protect user privacy. Always confirm before showing sensitive data. Avoid sounding like a bot."""

    def mask_email(self, email: str) -> str:
        """Mask email address for privacy protection."""
        if not email or '@' not in email:
            return email
        
        try:
            local, domain_parts = email.split('@', 1)
            
            if '.' in domain_parts:
                domain, tld = domain_parts.rsplit('.', 1)
            else:
                domain = domain_parts
                tld = ''
            
            if len(local) <= 4:
                masked_local = local
            else:
                masked_local = local[:2] + '****' + local[-2:]
            
            if len(domain) <= 4:
                masked_domain = domain
            else:
                masked_domain = domain[:2] + '***' + domain[-2:]
            
            if tld:
                masked_email = f"{masked_local}@{masked_domain}.{tld}"
            else:
                masked_email = f"{masked_local}@{masked_domain}"
            
            return masked_email
            
        except Exception as e:
            logger.error(f"Error masking email {email}: {e}")
            return email

    def call_llm(self, user_message: str, conversation_state: Dict = None,
                 customer_data: Dict = None, billing_result: Dict = None,
                 conversation_history: List[Dict] = None) -> Dict:
        """Call the Azure OpenAI LLM to process user message and return structured response."""
        if not self.ai_enabled or not self.client:
            return self._fallback_response(user_message, customer_data, billing_result)
        
        try:
            context_parts = []
            
            if conversation_history and len(conversation_history) > 0:
                recent_history = conversation_history[-5:]
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
                if conversation_state.get("pending_billing_confirmation"):
                    state_info += f"- PENDING BILLING CONFIRMATION: Waiting for user to confirm before showing billing data\n"
                if conversation_state.get("pending_fault_confirmation"):
                    state_info += f"- PENDING FAULT CONFIRMATION: Waiting for user to confirm before logging fault\n"
                context_parts.append(state_info)
            
            if customer_data:
                context_parts.append(f"CUSTOMER DATA: {json.dumps(customer_data)}")
            
            if billing_result:
                context_parts.append(f"BILLING STATUS: {json.dumps(billing_result)}")
            
            context = "\n".join(context_parts) if context_parts else "No additional context."
            
            user_prompt = f"""Customer message: "{user_message}"

{context}

Remember: Be NATURAL. Validate account exists. For billing, CONFIRM account details first. MASK EMAIL in confirmations. Return JSON only."""
            
            response = self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if "intent" not in result or "reply" not in result:
                raise ValueError("LLM response missing required fields")
            
            if "required_data" not in result:
                result["required_data"] = []
            
            logger.info(f"LLM detected intent: {result['intent']}")
            return result
            
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI LLM: {e}", exc_info=True)
            return self._fallback_response(user_message, customer_data, billing_result)

    def _fallback_response(self, user_message: str, customer_data: Dict = None,
                           billing_result: Dict = None) -> Dict:
        """Fallback pattern-matching based response when LLM is unavailable."""
        message_lower = user_message.lower().strip()
        
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return {
                "intent": "Greeting",
                "reply": "Hi! I'm here to help with your BEDC account. What can I assist you with today?",
                "required_data": []
            }
        
        if any(word in message_lower for word in ['bill', 'billing', 'charge', 'overcharge', 'nerc', 'cap']):
            if billing_result and customer_data:
                return {
                    "intent": "BillingConfirmation",
                    "reply": "Requesting confirmation before showing billing data",
                    "required_data": []
                }
            else:
                return {
                    "intent": "Billing",
                    "reply": "Could you share your account number so I can look into your billing?",
                    "required_data": ["account_number"]
                }
        
        if any(word in message_lower for word in ['meter', 'prepaid', 'map', 'apply']):
            return {
                "intent": "Metering",
                "reply": "You can apply for a prepaid meter at https://imaap.beninelectric.com:55682/",
                "required_data": []
            }
        
        if any(word in message_lower for word in ['fault', 'outage', 'no power', 'blackout']):
            return {
                "intent": "Fault",
                "reply": "I'm sorry about the outage. Could you share your account number?",
                "required_data": ["account_number", "email"]
            }
        
        return {
            "intent": "FAQ",
            "reply": "I can help with billing, meters, or fault reports. What do you need assistance with?",
            "required_data": []
        }

    def _is_affirmative(self, message: str) -> bool:
        """Use LLM to detect if message is an affirmative response."""
        if not message:
            return False
        
        # If LLM is not available, use simple keyword matching
        if not self.ai_enabled or not self.client:
            message_lower = message.lower().strip()
            simple_affirmatives = ['yes', 'y', 'yeah', 'yep', 'ok', 'correct', 'right']
            return any(word in message_lower for word in simple_affirmatives)
        
        try:
            prompt = f"""Is the following message an affirmative/positive response (like "yes", "correct", "that's right", "it is my account", etc.)?

Message: "{message}"

Respond with ONLY "YES" or "NO"."""
            
            response = self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": "You are a classification assistant. Respond with only YES or NO."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            is_affirmative = result == "YES"
            
            logger.info(f"LLM affirmative detection for '{message}': {is_affirmative}")
            return is_affirmative
            
        except Exception as e:
            logger.error(f"Error in LLM affirmative detection: {e}")
            # Fallback to simple keyword matching
            message_lower = message.lower().strip()
            simple_affirmatives = ['yes', 'y', 'yeah', 'yep', 'ok', 'correct', 'right']
            return any(word in message_lower for word in simple_affirmatives)

    def _is_negative(self, message: str) -> bool:
        """Use LLM to detect if message is a negative response."""
        if not message:
            return False
        
        # If LLM is not available, use simple keyword matching
        if not self.ai_enabled or not self.client:
            message_lower = message.lower().strip()
            simple_negatives = ['no', 'n', 'nope', 'wrong', 'incorrect', 'not']
            return any(word in message_lower for word in simple_negatives)
        
        try:
            prompt = f"""Is the following message a negative/rejection response (like "no", "wrong", "not correct", "that's not my account", etc.)?

Message: "{message}"

Respond with ONLY "YES" or "NO"."""
            
            response = self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": "You are a classification assistant. Respond with only YES or NO."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            is_negative = result == "YES"
            
            logger.info(f"LLM negative detection for '{message}': {is_negative}")
            return is_negative
            
        except Exception as e:
            logger.error(f"Error in LLM negative detection: {e}")
            # Fallback to simple keyword matching
            message_lower = message.lower().strip()
            simple_negatives = ['no', 'n', 'nope', 'wrong', 'incorrect', 'not']
            return any(word in message_lower for word in simple_negatives)

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
        Generate AI response with account validation and billing confirmation modal.

        Returns:
            Tuple[str, str, Dict]: (response, intent, updated_state)
        """
        if not user_message or not isinstance(user_message, str):
            return ("I'm sorry, I didn't receive a valid message. How can I help you?", "unknown", {})
        
        if session_state is None:
            session_state = {}
        
        pending_billing_confirmation = session_state.get("pending_billing_confirmation", False)
        pending_fault_confirmation = session_state.get("pending_fault_confirmation", False)
        
        # CRITICAL FIX: Check if the last message was a confirmation request
        if conversation_history and len(conversation_history) > 0:
            last_exchange = conversation_history[-1]
            last_intent = last_exchange.get("intent", "")
            
            # If last intent was BillingConfirmation and user is responding, we're in confirmation mode
            if last_intent == "BillingConfirmation":
                pending_billing_confirmation = True
                logger.info("Detected BillingConfirmation from last exchange - setting pending_billing_confirmation=True")
            
            # Same for fault confirmation
            if last_intent == "FaultConfirmation":
                pending_fault_confirmation = True
                logger.info("Detected FaultConfirmation from last exchange - setting pending_fault_confirmation=True")
        
        # Log state for debugging
        logger.info(f"Current state flags: pending_billing={pending_billing_confirmation}, pending_fault={pending_fault_confirmation}")
        
        # Handle billing confirmation FIRST (before calling LLM)
        if pending_billing_confirmation:
            logger.info(f"[INFO] BILLING CONFIRMATION MODE: Processing user response")
            if self._is_affirmative(user_message):
                logger.info(f"Affirmative detected! Proceeding to show billing data")
                # Get fresh billing data
                account_num = session_state.get("account_number")
                logger.info(f"Account number from session_state: {account_num}")
                
                if account_num:
                    logger.info(f"Account number exists, fetching data...")
                    # Fetch customer data and billing result
                    customer_data = self.data_manager.get_customer_by_account(account_num)
                    logger.info(f"Customer data retrieved: {customer_data is not None}")
                    if customer_data:
                        billing_result = self.data_manager.check_billing_status(account_num)
                        logger.info(f"Billing result retrieved: {billing_result is not None}")
                        
                        if billing_result:
                            customer = billing_result.get('customer_data', {})
                            feeder = customer.get('feeder', 'your')
                            is_metered = customer.get('metered', False)
                            
                            if billing_result['status'] == 'within_cap':
                                reply = f"Your bill of ₦{billing_result['bill_amount']:,} is within the ₦{billing_result['nerc_cap']:,} NERC cap for {feeder} feeder. "
                                if not is_metered:
                                    reply += "Since you're an unmetered customer, your billing follows the approved methodology. "
                                    reply += "For more accurate billing, apply for a prepaid meter at https://imaap.bedinelectric.com:55682/"
                            else:
                                reply = f"I sincerely apologize for this error. Your bill of ₦{billing_result['bill_amount']:,} exceeds the ₦{billing_result['nerc_cap']:,} NERC cap by ₦{billing_result['difference']:,}. "
                                reply += "We'll adjust it within one billing cycle."
                                if not is_metered:
                                    reply += " Get a prepaid meter for more accurate billing at https://imaap.bedinelectric.com:55682/"
                            
                            logger.info(f"Showing billing data for account {account_num}")
                            return (reply, "Billing", {
                                "billing_data": {},
                                "pending_billing_confirmation": False,
                                "billing_checked": True,
                                "account_number": account_num
                            })
            
            elif self._is_negative(user_message):
                return ("No problem. What would you like to update? Please provide the correct account number.", "Billing", {
                    "pending_billing_confirmation": False,
                    "billing_data": {},
                    "account_number": None
                })
            
            else:
                return ("I didn't quite catch that. Please reply 'Yes' to proceed or 'No' to update.", "BillingConfirmation", session_state)
        
        # Handle fault confirmation
        if pending_fault_confirmation:
            if self._is_affirmative(user_message):
                fault_data = session_state.get("fault_data", {})
                
                success = self.data_manager.save_fault_report(
                    fault_data["phone_number"],
                    fault_data["account_number"],
                    fault_data["email"],
                    fault_data.get("fault_description", "Power outage reported")
                )
                
                if success:
                    masked_email = self.mask_email(fault_data['email'])
                    
                    reply = "Fault report logged successfully!\n\n"
                    reply += f"Reference: FR-{fault_data['account_number']}-{datetime.now().strftime('%Y%m%d')}\n"
                    reply += f"Email: {masked_email}\n\n"
                    reply += "Our technical team will contact you within 24-48 hours. Thanks for your patience."
                    
                    return (reply, "Fault", {
                        "fault_data": {},
                        "pending_fault_confirmation": False
                    })
            
            elif self._is_negative(user_message):
                return ("No problem. What would you like to update?", "Fault", {
                    "pending_fault_confirmation": False,
                    "fault_data": {}
                })
            
            else:
                return ("Please reply 'Yes' to confirm or 'No' to update.", "FaultConfirmation", session_state)
        
        # Extract and validate account
        saved_account_number = session_state.get("account_number")
        account_number = self.extract_account_number(user_message)
        email = self.extract_email(user_message)
        
        # Check if user typed something that looks like an account but isn't valid (e.g., "12333")
        potential_account = re.search(r'\b(\d{5,6})\b', user_message)
        if potential_account and not account_number:
            # User typed numbers that look like account but don't match 10XXXX pattern
            invalid_num = potential_account.group(1)
            logger.warning(f"Invalid account number format detected: {invalid_num}")
            error_reply = f"I couldn't find an account associated with the number you provided. Please check and confirm:\n\n"
            error_reply += f"• Is the account number correct? (It should be 6 digits starting with '10')\n"
            error_reply += f"• If you don't have an account yet, visit our office at Ring Road, Benin City (Monday-Friday, 8AM-4PM) to create one. Bring valid ID and proof of address.\n\n"
            error_reply += f"You can also provide a different account number if there was a typo."
            
            return (error_reply, "AccountNotFound", {})
        
        # If account number doesn't match pattern (10XXXX), reject it immediately
        if account_number and not re.match(r'^10\d{4}$', account_number):
            logger.warning(f"Invalid account number format: {account_number}")
            error_reply = f"I couldn't find an account associated with the number you provided. Please check and confirm:\n\n"
            error_reply += f"• Is the account number correct? (It should be 6 digits starting with '10')\n"
            error_reply += f"• If you don't have an account yet, visit our office at Ring Road, Benin City (Monday-Friday, 8AM-4PM) to create one. Bring valid ID and proof of address.\n\n"
            error_reply += f"You can also provide a different account number if there was a typo."
            
            return (error_reply, "AccountNotFound", {})
        
        if not account_number and saved_account_number:
            account_number = saved_account_number
        
        customer_data = None
        billing_result = None
        customer_email = None
        account_exists = False
        
        if account_number:
            customer_data = self.data_manager.get_customer_by_account(account_number)
            if customer_data:
                account_exists = True
                billing_result = self.data_manager.check_billing_status(account_number)
                customer_email = customer_data.get("email")
                logger.info(f"[SUCCESS] Account {account_number} found")
            else:
                logger.warning(f"[ERROR] Account {account_number} NOT FOUND")
                error_reply = f"I couldn't find an account associated with the number you provided. Please check and confirm:\n\n"
                error_reply += f"• Is the account number correct? (It should be 6 digits starting with '10')\n"
                error_reply += f"• If you don't have an account yet, visit our office at Ring Road, Benin City (Monday-Friday, 8AM-4PM) to create one. Bring valid ID and proof of address.\n\n"
                error_reply += f"You can also provide a different account number if there was a typo."
                
                return (error_reply, "AccountNotFound", {})
        
        if not email and customer_email:
            email = customer_email
        
        conversation_state = {
            "phone_number": phone_number,
            "user_name": user_name,
            "has_account_number": bool(account_number),
            "has_email": bool(email),
            "email_from_database": bool(customer_email),
            "session_data": session_state,
            "saved_account_number": saved_account_number,
            "pending_billing_confirmation": pending_billing_confirmation,
            "pending_fault_confirmation": pending_fault_confirmation,
            "account_exists": account_exists
        }
        
        if account_number and account_exists:
            conversation_state["account_number"] = account_number
        
        llm_response = self.call_llm(
            user_message,
            conversation_state,
            customer_data,
            billing_result,
            conversation_history
        )
        
        intent = llm_response.get("intent", "unknown")
        reply = llm_response.get("reply", "I'm here to help. What can I assist you with?")
        
        state_update = {}
        
        if account_number and account_exists:
            state_update["account_number"] = account_number
        
        # BILLING CONFIRMATION MODAL
        if intent == "Billing" and account_number and account_exists and billing_result:
            masked_email = self.mask_email(customer_email) if customer_email else "Not on file"
            
            confirmation_message = f"Let me check your account. Is this correct?\n\n"
            confirmation_message += f"Account: {account_number}\n"
            confirmation_message += f"Email: {masked_email}\n\n"
            confirmation_message += "Reply 'Yes' to proceed or 'No' to update."
            
            state_update["billing_data"] = billing_result
            state_update["pending_billing_confirmation"] = True  # CRITICAL: Set this flag
            state_update["account_number"] = account_number
            
            logger.info(f"[INFO] Setting pending_billing_confirmation=True for account {account_number}")
            
            return (confirmation_message, "BillingConfirmation", state_update)
        
        # If LLM returned BillingConfirmation but account doesn't exist, override with AccountNotFound
        elif intent == "BillingConfirmation" and account_number and not account_exists:
            logger.warning(f"LLM hallucinated BillingConfirmation for non-existent account {account_number}")
            error_reply = f"I couldn't find an account associated with the number you provided. Please check and confirm:\n\n"
            error_reply += f"• Is the account number correct? (It should be 6 digits starting with '10')\n"
            error_reply += f"• If you don't have an account yet, visit our office at Ring Road, Benin City (Monday-Friday, 8AM-4PM) to create one. Bring valid ID and proof of address.\n\n"
            error_reply += f"You can also provide a different account number if there was a typo."
            
            return (error_reply, "AccountNotFound", {})
        
        # FAULT CONFIRMATION
        elif intent == "Fault":
            fault_data = session_state.get("fault_data", {})
            
            if account_number and account_exists:
                fault_data["account_number"] = account_number
            if email:
                fault_data["email"] = email
            if phone_number:
                fault_data["phone_number"] = phone_number
            if not fault_data.get("fault_description"):
                fault_data["fault_description"] = user_message
            
            if fault_data.get("account_number") and fault_data.get("email") and fault_data.get("phone_number"):
                masked_email = self.mask_email(fault_data['email'])
                
                confirmation_message = f"Just to confirm - is this your account?\n\n"
                confirmation_message += f"Account: {fault_data['account_number']}\n"
                confirmation_message += f"Email: {masked_email}\n\n"
                confirmation_message += "Reply 'Yes' to confirm or 'No' to update."
                
                state_update["fault_data"] = fault_data
                state_update["pending_fault_confirmation"] = True
                
                return (confirmation_message, "FaultConfirmation", state_update)
            else:
                state_update["fault_data"] = fault_data
        
        return (reply, intent, state_update)