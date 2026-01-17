import logging
from typing import Dict, Any
from handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class FAQHandler(BaseHandler):
    """Handles FAQ interactions with categorized questions."""

    def __init__(self, config, session_manager, data_manager, whatsapp_service):
        super().__init__(config, session_manager, data_manager, whatsapp_service)
        
        # FAQ Categories
        self.faq_categories = {
            "billing": {
                "title": "💰 Billing Questions",
                "questions": {
                    "1": {
                        "q": "What is NERC capping?",
                        "a": "NERC (Nigerian Electricity Regulatory Commission) capping is the maximum amount an unmetered customer can be charged monthly based on their feeder classification."
                    },
                    "2": {
                        "q": "Why is my bill higher than the NERC cap?",
                        "a": "Bills above the NERC cap are billing errors. We apologize for this. Your account will be reviewed and adjusted within one billing cycle."
                    },
                    "3": {
                        "q": "How is the NERC cap calculated?",
                        "a": "NERC caps are based on feeder classification and customer category. Unmetered customers are charged estimated consumption within these caps."
                    },
                    "4": {
                        "q": "What if I disagree with my bill?",
                        "a": "Provide your account number and we'll review it against NERC caps and adjust if there's an error."
                    }
                }
            },
            "metering": {
                "title": "⚡ Metering Questions",
                "questions": {
                    "1": {
                        "q": "How do I apply for a prepaid meter?",
                        "a": "Visit https://imaap.beninelectric.com:55682/ and follow the MAP enrollment process. You'll need your account number.\n\n📌 Resources:\n• Metering: https://beninelectric.com/metering/\n• E-Billing: https://beninelectric.com/e-billing/"
                    },
                    "2": {
                        "q": "What is MAP?",
                        "a": "MAP stands for Meter Asset Provider - a scheme where you purchase your prepaid meter directly from approved vendors."
                    },
                    "3": {
                        "q": "Can I use one account for multiple meters?",
                        "a": "No. Each meter location requires a separate postpaid account number. You cannot use one account for multiple meter applications."
                    },
                    "4": {
                        "q": "How long does meter installation take?",
                        "a": "After payment through MAP, installation is typically scheduled within 2-4 weeks."
                    },
                    "5": {
                        "q": "How much does a prepaid meter cost?",
                        "a": "Meter costs vary by type (single-phase vs three-phase). Visit https://imaap.beninelectric.com:55682/ for current pricing."
                    }
                }
            },
            "account": {
                "title": "📋 Account Questions",
                "questions": {
                    "1": {
                        "q": "I'm a new customer. How do I get a meter?",
                        "a": "You must first visit our office at Ring Road, Benin City to create a postpaid account. Then you can apply for a meter through MAP.\n\n📍 Bring: Valid ID, Proof of address, Utility bill (if available)"
                    },
                    "2": {
                        "q": "What are BEDC's office hours?",
                        "a": "Monday-Friday, 8:00 AM - 4:00 PM at Ring Road, Benin City."
                    },
                    "3": {
                        "q": "How do I update my contact information?",
                        "a": "Visit our office with valid ID to update your phone number, email, or address on your account."
                    },
                    "4": {
                        "q": "Can I switch from postpaid to prepaid?",
                        "a": "Yes! Enroll in MAP to get a prepaid meter. Your postpaid account will be closed once the meter is installed."
                    }
                }
            },
            "service": {
                "title": "🔧 Service Questions",
                "questions": {
                    "1": {
                        "q": "How do I report a power outage?",
                        "a": "Provide your account number, phone number, and email. We'll log your report and our technical team will respond within 24-48 hours."
                    },
                    "2": {
                        "q": "What areas does BEDC serve?",
                        "a": "BEDC serves Edo, Delta, Ondo, and Ekpoma areas with multiple feeders across these regions."
                    },
                    "3": {
                        "q": "What is my feeder?",
                        "a": "Your feeder is shown on your bill. It determines your NERC cap amount. Contact us with your account number to confirm."
                    }
                }
            },
            "payment": {
                "title": "💳 Payment Questions",
                "questions": {
                    "1": {
                        "q": "Can I pay my bill through WhatsApp?",
                        "a": "Currently, bill payments are not available via WhatsApp. Please visit our office or use bank channels."
                    },
                    "2": {
                        "q": "Can I get a meter if I have unpaid bills?",
                        "a": "Outstanding bills should be cleared before MAP enrollment. Contact our office for payment arrangements."
                    },
                    "3": {
                        "q": "How do I check my current bill?",
                        "a": "Provide your account number and we'll check your billing status immediately."
                    }
                }
            }
        }

    def handle_faq_state(self, state: Dict, message: str, original_message: str, session_id: str) -> Dict[str, Any]:
        """Handle FAQ navigation."""
        logger.info(f"FAQHandler: Handling message '{message}' for session {session_id}")
        
        # Check if user selected a category
        if message in self.faq_categories:
            return self._show_category_questions(state, session_id, message)
        
        # Check if user selected a specific question
        current_category = state.get("faq_category")
        if current_category and message.isdigit():
            return self._show_answer(state, session_id, current_category, message)
        
        # Check for back to categories or main menu
        if message == "back_to_categories":
            return self._show_categories(state, session_id)
        elif message == "back_to_main":
            return self.handle_back_to_main(state, session_id)
        
        # Default: show categories
        return self._show_categories(state, session_id)

    def _show_categories(self, state: Dict, session_id: str) -> Dict[str, Any]:
        """Display FAQ categories."""
        state["current_state"] = "faq"
        state["current_handler"] = "faq_handler"
        state["faq_category"] = None
        self.session_manager.update_session_state(session_id, state)
        
        message = "📚 **Frequently Asked Questions**\n\nPlease select a category:\n\n"
        buttons = []
        
        for key, category in self.faq_categories.items():
            message += f"{category['title']}\n"
            buttons.append({
                "id": key,
                "title": category['title'].replace("💰 ", "").replace("⚡ ", "").replace("📋 ", "").replace("🔧 ", "").replace("💳 ", "")[:20]
            })
        
        message += "\n\nType 'menu' to return to main menu."
        
        return self.whatsapp_service.create_interactive_message(
            session_id,
            message,
            buttons,
            "Select Category"
        )

    def _show_category_questions(self, state: Dict, session_id: str, category: str) -> Dict[str, Any]:
        """Display questions for a specific category."""
        category_data = self.faq_categories.get(category)
        if not category_data:
            return self._show_categories(state, session_id)
        
        state["faq_category"] = category
        self.session_manager.update_session_state(session_id, state)
        
        message = f"{category_data['title']}\n\n"
        buttons = []
        
        for q_id, question in category_data['questions'].items():
            message += f"{q_id}. {question['q']}\n"
            buttons.append({
                "id": q_id,
                "title": f"Q{q_id}: {question['q'][:15]}..."
            })
        
        # Add back button
        buttons.append({
            "id": "back_to_categories",
            "title": "← Back to Categories"
        })
        
        return self.whatsapp_service.create_interactive_message(
            session_id,
            message,
            buttons,
            "Select Question"
        )

    def _show_answer(self, state: Dict, session_id: str, category: str, question_id: str) -> Dict[str, Any]:
        """Display answer to a specific question."""
        category_data = self.faq_categories.get(category)
        if not category_data:
            return self._show_categories(state, session_id)
        
        question_data = category_data['questions'].get(question_id)
        if not question_data:
            return self._show_category_questions(state, session_id, category)
        
        message = f"**Q: {question_data['q']}**\n\n{question_data['a']}\n\n"
        message += "---\n\n"
        message += "Type a number for another question in this category,\n"
        message += "'back' for categories, or 'menu' for main menu."
        
        return self.whatsapp_service.create_text_message(session_id, message)

    def handle_back_to_main(self, state: Dict, session_id: str, message: str = "") -> Dict[str, Any]:
        """Return to AI chat from FAQ."""
        logger.info(f"FAQHandler: Returning to main from session {session_id}")
        
        state["current_state"] = "ai_chat"
        state["current_handler"] = "ai_handler"
        state["faq_category"] = None
        
        self.session_manager.update_session_state(session_id, state)
        
        return {
            "redirect": "ai_handler",
            "redirect_message": "initial_greeting",
            "additional_message": "How else can I help you?"
        }