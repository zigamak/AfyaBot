import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class AIService:
    """
    Handles AI-powered conversational interactions for BEDC WhatsApp Support Bot,
    focusing on billing complaints, metering inquiries, and fault reporting.
    """

    def __init__(self, config, data_manager):
        """Initialize AI Service with configuration and data manager."""
        self.data_manager = data_manager
        self.ai_enabled = True  # Always enabled for rule-based responses
        
        # Get OpenAI API key if available (for future LLM integration)
        try:
            if isinstance(config, dict):
                self.openai_api_key = config.get("openai_api_key")
            else:
                self.openai_api_key = getattr(config, 'OPENAI_API_KEY', os.getenv("OPENAI_API_KEY"))
        except:
            self.openai_api_key = None
        
        logger.info("AI Service initialized successfully for BEDC Support Bot")

    def detect_intent(self, message: str) -> str:
        """
        Detect user intent from message.

        Args:
            message (str): User's message

        Returns:
            str: Detected intent (greeting, billing, metering, fault, faq, unknown)
        """
        if not message or not isinstance(message, str):
            return "unknown"
        
        message_lower = message.lower().strip()
        
        # Greeting patterns
        greeting_patterns = [
            r'\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b',
            r'\b(start|begin|help)\b'
        ]
        
        # Billing patterns
        billing_patterns = [
            r'\b(bill|billing|charge|payment|invoice|overcharge|overbill)\b',
            r'\b(nerc|cap|capping|too high|expensive|unfair)\b',
            r'\b(complain|complaint|dispute|query)\b.*\b(bill|charge)',
            r'\b(my bill|bill is)\b'
        ]
        
        # Metering patterns
        metering_patterns = [
            r'\b(meter|prepaid|postpaid|map|meter asset provider)\b',
            r'\b(order.*meter|get.*meter|apply.*meter|request.*meter)\b',
            r'\b(want.*meter|need.*meter|installation)\b'
        ]
        
        # Fault patterns
        fault_patterns = [
            r'\b(fault|outage|power.*out|no.*power|no.*light|blackout)\b',
            r'\b(electricity.*gone|power.*cut|down|not working)\b',
            r'\b(report.*fault|report.*outage|problem.*supply)\b'
        ]
        
        # FAQ patterns
        faq_patterns = [
            r'\b(multiple.*meter|several.*meter|more than one)\b',
            r'\b(new customer|no account|open.*account|create.*account)\b',
            r'\b(how.*work|what.*is|tell.*about)\b'
        ]
        
        # Check patterns in priority order
        if any(re.search(pattern, message_lower) for pattern in greeting_patterns):
            return "greeting"
        elif any(re.search(pattern, message_lower) for pattern in billing_patterns):
            return "billing"
        elif any(re.search(pattern, message_lower) for pattern in metering_patterns):
            return "metering"
        elif any(re.search(pattern, message_lower) for pattern in fault_patterns):
            return "fault"
        elif any(re.search(pattern, message_lower) for pattern in faq_patterns):
            return "faq"
        
        return "unknown"

    def extract_account_number(self, message: str) -> Optional[str]:
        """Extract account number from message."""
        if not message:
            return None
        
        # Look for 6-digit account numbers starting with 10
        match = re.search(r'\b(10\d{4})\b', message)
        return match.group(1) if match else None

    def extract_email(self, message: str) -> Optional[str]:
        """Extract email from message."""
        if not message:
            return None
        
        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message)
        return match.group(0) if match else None

    def handle_greeting(self, user_name: str = "Customer") -> str:
        """Generate greeting response."""
        return f"""Hello {user_name}! Welcome to BEDC Customer Support. 🌟

I'm here to help you with:
📋 Billing inquiries and complaints
⚡ Meter applications (MAP enrollment)
🔧 Fault reporting and power outages
❓ General questions about our services

How may I assist you today?"""

    def handle_billing_inquiry(self, message: str, account_number: str = None) -> Tuple[str, bool, Optional[str]]:
        """
        Handle billing-related inquiries.

        Returns:
            Tuple[str, bool, Optional[str]]: (response, needs_account_number, detected_account)
        """
        # Check if account number is in message
        if not account_number:
            account_number = self.extract_account_number(message)
        
        if not account_number:
            return (
                "I understand you have a billing concern. To assist you properly, "
                "please provide your 6-digit account number (e.g., 100001).",
                True,
                None
            )
        
        # Check billing status
        result = self.data_manager.check_billing_status(account_number)
        
        if result["status"] == "not_found":
            return (
                f"I apologize, but I couldn't find account number {account_number} in our records. "
                "Please verify the account number and try again, or visit our office for assistance.",
                False,
                account_number
            )
        
        customer = result["customer_data"]
        bill = result["bill_amount"]
        cap = result["nerc_cap"]
        
        if result["status"] == "within_cap":
            response = f"""Dear {customer['customer_name']},

I've reviewed your account ({account_number}) on {customer['feeder']} feeder.

📊 Current Bill: ₦{bill:,}
📈 NERC Cap: ₦{cap:,}
✅ Status: WITHIN regulatory limits

Your billing follows the NERC-approved methodology for unmetered customers. 

💡 To avoid estimated billing, I recommend enrolling in our Meter Asset Provider (MAP) scheme for a prepaid meter.

Visit: https://bedc.com/order-meter to apply online.

Is there anything else I can help you with?"""
        else:
            diff = result["difference"]
            response = f"""Dear {customer['customer_name']},

I sincerely apologize for the inconvenience regarding your billing.

📊 Current Bill: ₦{bill:,}
📈 NERC Cap: ₦{cap:,}
⚠️ Status: ₦{diff:,} ABOVE the regulatory cap

I acknowledge this discrepancy. Our billing team will review and adjust your account within ONE billing cycle.

💡 To prevent future billing issues, I strongly encourage you to enroll in our MAP scheme for a prepaid meter: https://bedc.com/order-meter

Would you like me to log your MAP application interest?"""
        
        return (response, False, account_number)

    def handle_metering_inquiry(self, message: str) -> Tuple[str, str]:
        """
        Handle meter-related inquiries.

        Returns:
            Tuple[str, str]: (response, next_action)
        """
        message_lower = message.lower()
        
        # Check if user has account number
        account_number = self.extract_account_number(message)
        
        if "new customer" in message_lower or "no account" in message_lower or "don't have account" in message_lower:
            return (
                """For new customers without a postpaid account:

📍 You'll need to visit our office to create an account first.

**BEDC Customer Service Center**
📌 Address: Ring Road, Benin City
⏰ Hours: Monday-Friday, 8:00 AM - 4:00 PM

Please bring:
- Valid ID
- Proof of address
- Utility bill (if available)

After your account is created, you can apply for a prepaid meter through MAP.

Any other questions?""",
                "office_visit_required"
            )
        
        if "multiple" in message_lower or "several" in message_lower or "more than one" in message_lower:
            return (
                """⚠️ Important Policy:

One postpaid account number CANNOT be used for multiple meter applications.

Each meter location requires a separate account. If you need meters for multiple properties, you must:
1. Create separate accounts for each location
2. Apply for MAP enrollment individually

Visit our office for assistance with multiple accounts.

How else can I help?""",
                "policy_clarification"
            )
        
        # Standard MAP enrollment guidance
        return (
            """**Meter Asset Provider (MAP) Enrollment Guide** ⚡

To get your prepaid meter:

1️⃣ Visit: https://bedc.com/order-meter
2️⃣ Click "Order a Meter"
3️⃣ Fill in your account details
4️⃣ Complete payment
5️⃣ Installation scheduled within 2-4 weeks

📋 Requirements:
- Valid BEDC account number
- Correct contact information
- Payment for meter cost

Need help with the application process?""",
            "map_enrollment"
        )

    def handle_fault_report(self, message: str, phone_number: str, 
                           collected_data: Dict = None) -> Tuple[str, bool, Dict]:
        """
        Handle fault/outage reporting with data collection.

        Returns:
            Tuple[str, bool, Dict]: (response, needs_more_data, collected_data)
        """
        if collected_data is None:
            collected_data = {}
        
        # Extract information from message
        account_number = collected_data.get("account_number") or self.extract_account_number(message)
        email = collected_data.get("email") or self.extract_email(message)
        
        # Update collected data
        if account_number:
            collected_data["account_number"] = account_number
        if email:
            collected_data["email"] = email
        if not collected_data.get("phone_number"):
            collected_data["phone_number"] = phone_number
        if not collected_data.get("fault_description"):
            collected_data["fault_description"] = message
        
        # Check if we have all required information
        missing = []
        if not collected_data.get("account_number"):
            missing.append("account number")
        if not collected_data.get("email"):
            missing.append("email address")
        
        if missing:
            return (
                f"""I understand you're experiencing a power outage. I sincerely apologize for this inconvenience.

To log your fault report, I need your:
{chr(10).join(f'- {item.title()}' for item in missing)}

Please provide the missing information.""",
                True,
                collected_data
            )
        
        # Save fault report
        success = self.data_manager.save_fault_report(
            collected_data["phone_number"],
            collected_data["account_number"],
            collected_data["email"],
            collected_data.get("fault_description", "Power outage reported")
        )
        
        if success:
            return (
                f"""✅ Fault Report Logged Successfully

📋 Reference: FR-{collected_data['account_number']}-{datetime.now().strftime('%Y%m%d')}
📞 Phone: {collected_data['phone_number']}
📧 Email: {collected_data['email']}

Our technical team will investigate and contact you within 24-48 hours.

We apologize for the inconvenience and appreciate your patience.

Is there anything else I can help you with?""",
                False,
                {}
            )
        else:
            return (
                "I apologize, but there was an error logging your fault report. "
                "Please try again or contact our office directly.",
                False,
                {}
            )

    def generate_response(self, user_message: str, conversation_history: List[Dict] = None,
                         phone_number: str = None, user_name: str = None,
                         session_state: Dict = None) -> Tuple[str, str, Dict]:
        """
        Generate AI response using intent detection and appropriate handler.

        Returns:
            Tuple[str, str, Dict]: (response, intent, updated_state)
        """
        if not user_message or not isinstance(user_message, str):
            return ("I'm sorry, I didn't receive a valid message. How can I help you?", "unknown", {})
        
        if session_state is None:
            session_state = {}
        
        # Detect intent
        intent = self.detect_intent(user_message)
        logger.info(f"Detected intent: {intent} for message: {user_message[:50]}")
        
        # Handle based on intent
        if intent == "greeting":
            response = self.handle_greeting(user_name or "Customer")
            return (response, intent, {})
        
        elif intent == "billing":
            response, needs_account, account = self.handle_billing_inquiry(user_message)
            state_update = {"needs_account_number": needs_account}
            if account:
                state_update["account_number"] = account
            return (response, intent, state_update)
        
        elif intent == "metering":
            response, next_action = self.handle_metering_inquiry(user_message)
            return (response, intent, {"next_action": next_action})
        
        elif intent == "fault":
            collected_data = session_state.get("fault_data", {})
            response, needs_more, updated_data = self.handle_fault_report(
                user_message, phone_number, collected_data
            )
            state_update = {"fault_data": updated_data} if needs_more else {}
            return (response, intent, state_update)
        
        elif intent == "faq":
            # Handle FAQ intent
            response = """I'm here to help! Here are some common questions:

**Multiple Meters**: One account cannot be used for multiple meter applications
**New Customers**: Visit our office to create an account first
**MAP Enrollment**: Visit https://bedc.com/order-meter to apply for a prepaid meter
**Billing Issues**: Provide your account number to check your bill status

What specific question do you have?"""
            return (response, intent, {})
        
        else:
            # Default response for unknown intents
            return (
                """I'm here to help with:
- Billing inquiries and complaints
- Meter applications (MAP enrollment)
- Fault reports and power outages
- General BEDC service questions

Please let me know how I can assist you!""",
                "unknown",
                {}
            )