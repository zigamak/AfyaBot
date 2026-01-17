import logging
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DataManager:
    """Manages data interactions for BEDC WhatsApp Bot using JSON file storage."""

    def __init__(self, config=None):
        """
        Initialize DataManager with JSON file paths.

        Args:
            config: Configuration object or dictionary (optional for JSON-based storage)
        """
        # Set up data directory
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Define file paths
        self.customer_data_file = self.data_dir / "customer_data.json"
        self.conversations_file = self.data_dir / "conversations.json"
        self.fault_reports_file = self.data_dir / "fault_reports.json"
        self.map_applications_file = self.data_dir / "map_applications.json"
        
        # Initialize data structures
        self.customer_data = self._load_json(self.customer_data_file, {"customers": [], "feeders": {}})
        self.conversations = self._load_json(self.conversations_file, {})
        self.fault_reports = self._load_json(self.fault_reports_file, [])
        self.map_applications = self._load_json(self.map_applications_file, [])
        
        logger.info("DataManager initialized with JSON file storage")
        logger.info(f"Loaded {len(self.customer_data.get('customers', []))} customer records")

    def _load_json(self, filepath: Path, default: Any) -> Any:
        """Load JSON data from file or return default if file doesn't exist."""
        try:
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create file with default data
                self._save_json(filepath, default)
                return default
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return default

    def _save_json(self, filepath: Path, data: Any) -> bool:
        """Save data to JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving to {filepath}: {e}")
            return False

    def get_customer_by_account(self, account_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve customer information by account number.

        Args:
            account_number (str): Customer's account number

        Returns:
            Optional[Dict[str, Any]]: Customer data or None if not found
        """
        try:
            customers = self.customer_data.get("customers", [])
            for customer in customers:
                if customer.get("account_number") == account_number:
                    logger.info(f"Found customer: {account_number}")
                    return customer
            logger.info(f"Customer not found: {account_number}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving customer {account_number}: {e}")
            return None

    def check_billing_status(self, account_number: str) -> Dict[str, Any]:
        """
        Check if customer's billing is within NERC cap.

        Args:
            account_number (str): Customer's account number

        Returns:
            Dict with keys: status, customer_data, bill_amount, nerc_cap, difference
        """
        try:
            customer = self.get_customer_by_account(account_number)
            if not customer:
                return {
                    "status": "not_found",
                    "message": "Account number not found in our records"
                }
            
            bill_amount = customer.get("bill_amount", 0)
            nerc_cap = customer.get("nerc_cap", 0)
            difference = bill_amount - nerc_cap
            
            if bill_amount <= nerc_cap:
                status = "within_cap"
            else:
                status = "above_cap"
            
            return {
                "status": status,
                "customer_data": customer,
                "bill_amount": bill_amount,
                "nerc_cap": nerc_cap,
                "difference": difference
            }
        except Exception as e:
            logger.error(f"Error checking billing status for {account_number}: {e}")
            return {"status": "error", "message": str(e)}

    def save_conversation(self, phone_number: str, session_id: str, user_message: str, 
                         assistant_response: str, intent: str = None):
        """
        Save a conversation entry.

        Args:
            phone_number (str): User's phone number
            session_id (str): Session identifier
            user_message (str): User's message
            assistant_response (str): Bot's response
            intent (str): Detected intent (optional)
        """
        try:
            if phone_number not in self.conversations:
                self.conversations[phone_number] = []
            
            conversation_entry = {
                "session_id": session_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "intent": intent,
                "timestamp": datetime.now().isoformat()
            }
            
            self.conversations[phone_number].append(conversation_entry)
            
            # Keep only last 50 conversations per user
            if len(self.conversations[phone_number]) > 50:
                self.conversations[phone_number] = self.conversations[phone_number][-50:]
            
            self._save_json(self.conversations_file, self.conversations)
            logger.info(f"Saved conversation for {phone_number}, intent: {intent}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

    def get_conversation_history(self, phone_number: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a phone number.

        Args:
            phone_number (str): User's phone number
            limit (int): Maximum number of conversations to return

        Returns:
            List[Dict[str, Any]]: List of conversation entries
        """
        try:
            conversations = self.conversations.get(phone_number, [])
            return conversations[-limit:] if conversations else []
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    def save_fault_report(self, phone_number: str, account_number: str, 
                         email: str, fault_description: str) -> bool:
        """
        Save a fault/outage report.

        Args:
            phone_number (str): User's phone number
            account_number (str): Customer's account number
            email (str): User's email
            fault_description (str): Description of the fault

        Returns:
            bool: True if saved successfully
        """
        try:
            fault_report = {
                "phone_number": phone_number,
                "account_number": account_number,
                "email": email,
                "fault_description": fault_description,
                "status": "pending",
                "timestamp": datetime.now().isoformat()
            }
            
            self.fault_reports.append(fault_report)
            self._save_json(self.fault_reports_file, self.fault_reports)
            logger.info(f"Saved fault report for account {account_number}")
            return True
        except Exception as e:
            logger.error(f"Error saving fault report: {e}")
            return False

    def save_map_application(self, phone_number: str, account_number: str, 
                           customer_name: str, email: str = None) -> bool:
        """
        Save a MAP (Meter Asset Provider) application request.

        Args:
            phone_number (str): User's phone number
            account_number (str): Customer's account number
            customer_name (str): Customer's name
            email (str): User's email (optional)

        Returns:
            bool: True if saved successfully
        """
        try:
            map_application = {
                "phone_number": phone_number,
                "account_number": account_number,
                "customer_name": customer_name,
                "email": email,
                "status": "pending",
                "timestamp": datetime.now().isoformat()
            }
            
            self.map_applications.append(map_application)
            self._save_json(self.map_applications_file, self.map_applications)
            logger.info(f"Saved MAP application for account {account_number}")
            return True
        except Exception as e:
            logger.error(f"Error saving MAP application: {e}")
            return False

    def get_feeder_info(self, feeder_name: str) -> Optional[Dict[str, Any]]:
        """
        Get feeder information including NERC cap.

        Args:
            feeder_name (str): Name of the feeder

        Returns:
            Optional[Dict[str, Any]]: Feeder information or None
        """
        try:
            feeders = self.customer_data.get("feeders", {})
            return feeders.get(feeder_name)
        except Exception as e:
            logger.error(f"Error retrieving feeder info: {e}")
            return None

    def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics data for dashboard/reporting.

        Returns:
            Dict with analytics metrics
        """
        try:
            total_customers = len(self.customer_data.get("customers", []))
            total_conversations = sum(len(convs) for convs in self.conversations.values())
            total_fault_reports = len(self.fault_reports)
            total_map_applications = len(self.map_applications)
            
            # Count billing issues
            above_cap_count = 0
            for customer in self.customer_data.get("customers", []):
                if customer.get("bill_amount", 0) > customer.get("nerc_cap", 0):
                    above_cap_count += 1
            
            return {
                "total_customers": total_customers,
                "total_conversations": total_conversations,
                "total_fault_reports": total_fault_reports,
                "total_map_applications": total_map_applications,
                "customers_above_cap": above_cap_count,
                "unmetered_customers": total_customers  # All are unmetered in sample data
            }
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {}

    def close(self):
        """Save all data and cleanup."""
        try:
            self._save_json(self.conversations_file, self.conversations)
            self._save_json(self.fault_reports_file, self.fault_reports)
            self._save_json(self.map_applications_file, self.map_applications)
            logger.info("DataManager closed and all data saved")
        except Exception as e:
            logger.error(f"Error closing DataManager: {e}")