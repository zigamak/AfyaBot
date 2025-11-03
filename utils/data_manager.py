import logging
import os
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class DataManager:
    """Manages database interactions for AfyaBot, including kit requests, test results, and conversations."""

    def __init__(self, config=None):
        """
        Initialize DataManager with Supabase client using configuration or environment variables.

        Args:
            config: Configuration object or dictionary with Supabase credentials.
        """
        load_dotenv()
        
        # Get Supabase credentials from config or environment
        if isinstance(config, dict):
            self.supabase_url = config.get("supabase_url")
            self.supabase_key = config.get("supabase_service_key")
        else:
            self.supabase_url = getattr(config, 'SUPABASE_URL', os.getenv("SUPABASE_URL"))
            self.supabase_key = getattr(config, 'SUPABASE_SERVICE_KEY', os.getenv("SUPABASE_SERVICE_KEY"))
        
        if not self.supabase_url or not self.supabase_key:
            logger.error("Supabase URL or Service Key not found in environment variables or config")
            raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

        try:
            # Initialize Supabase client
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized successfully")
            self._verify_tables()
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
            raise

    def _verify_tables(self):
        """
        Verify that required tables exist in Supabase by attempting a simple query.
        Note: Supabase tables must be pre-created in the dashboard or via SQL.
        """
        required_tables = ["kit_requests", "test_results", "conversations"]
        for table in required_tables:
            try:
                # Attempt a lightweight query to check table existence
                self.supabase_client.table(table).select("id").limit(0).execute()
                logger.info(f"Table {table} verified in Supabase")
            except Exception as e:
                logger.warning(f"Table {table} may not exist or is inaccessible: {e}. Please create it in the Supabase dashboard.")
                logger.info(f"Required schema for {table}: {self._get_table_schema(table)}")

    def _get_table_schema(self, table_name: str) -> str:
        """Return the SQL schema for a given table for logging purposes."""
        schemas = {
            "kit_requests": """
                CREATE TABLE kit_requests (
                    id SERIAL PRIMARY KEY,
                    phone_number VARCHAR(20) UNIQUE NOT NULL,
                    name VARCHAR(100),
                    location VARCHAR(100),
                    request_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """,
            "test_results": """
                CREATE TABLE test_results (
                    id SERIAL PRIMARY KEY,
                    phone_number VARCHAR(20) UNIQUE NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """,
            "conversations": """
                CREATE TABLE conversations (
                    id SERIAL PRIMARY KEY,
                    phone_number VARCHAR(20) NOT NULL,
                    session_id VARCHAR(100),
                    user_message TEXT,
                    assistant_response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX idx_conversations_phone_number ON conversations(phone_number);
                CREATE INDEX idx_conversations_timestamp ON conversations(timestamp);
            """
        }
        return schemas.get(table_name, "Unknown table")

    def _truncate_string(self, value: str, max_length: int, field_name: str) -> str:
        """Helper method to safely truncate strings to fit database constraints."""
        if value and isinstance(value, str) and len(value) > max_length:
            truncated = value[:max_length]
            logger.warning(f"Truncated {field_name} from '{value}' to '{truncated}' (max length: {max_length})")
            return truncated
        return value

    def _validate_phone_number(self, phone_number: str) -> bool:
        """Validate phone number is not empty and within length constraints."""
        if not phone_number or not isinstance(phone_number, str):
            logger.error("Phone number cannot be empty or None")
            return False
        if len(phone_number) > 20:
            logger.error(f"Phone number '{phone_number}' exceeds 20 characters")
            return False
        return True

    def save_conversation(self, phone_number: str, session_id: str, user_message: str, assistant_response: str):
        """
        Save a conversation entry for a given phone number.

        Args:
            phone_number (str): User's phone number
            session_id (str): Session identifier
            user_message (str): User's message
            assistant_response (str): Assistant's response
        """
        try:
            if not self._validate_phone_number(phone_number):
                raise ValueError("Invalid phone number for saving conversation")
            
            # Validate and truncate fields
            phone_number = self._truncate_string(phone_number, 20, "phone_number")
            session_id = self._truncate_string(session_id, 100, "session_id") if session_id else None
            user_message = self._truncate_string(user_message, 1000, "user_message") if user_message else None
            assistant_response = self._truncate_string(assistant_response, 1000, "assistant_response") if assistant_response else None
            
            # Insert into conversations table
            self.supabase_client.table("conversations").insert({
                "phone_number": phone_number,
                "session_id": session_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "timestamp": datetime.now().isoformat()
            }).execute()
            logger.info(f"Saved conversation for phone_number: {phone_number}, session_id: {session_id}")
        except Exception as e:
            logger.error(f"Error saving conversation for phone_number={phone_number}, session_id={session_id}: {e}", exc_info=True)
            raise

    def get_conversation_history(self, phone_number: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a phone number.

        Args:
            phone_number (str): User's phone number
            limit (int): Maximum number of conversations to return (default: 10)

        Returns:
            List[Dict[str, Any]]: List of conversation entries
        """
        try:
            if not self._validate_phone_number(phone_number):
                raise ValueError("Invalid phone number for retrieving conversation history")
            
            phone_number = self._truncate_string(phone_number, 20, "phone_number")
            
            # Query conversations table
            response = self.supabase_client.table("conversations").select("*").eq("phone_number", phone_number).order("timestamp", desc=True).limit(limit).execute()
            return [
                {
                    "session_id": row["session_id"],
                    "user": row["user_message"],
                    "assistant": row["assistant_response"],
                    "timestamp": row["timestamp"]
                } for row in response.data
            ]
        except Exception as e:
            logger.error(f"Error retrieving conversation history for {phone_number}: {e}", exc_info=True)
            return []

    def add_kit_request(self, phone_number: str, name: str = None, location: str = None, request_details: str = None):
        """
        Add or update a kit request in the kit_requests table.

        Args:
            phone_number (str): User's phone number (required)
            name (str, optional): User's name
            location (str, optional): User's location
            request_details (str, optional): Additional request details
        """
        try:
            if not self._validate_phone_number(phone_number):
                raise ValueError("Invalid phone number for kit request")
            
            # Validate and truncate fields
            phone_number = self._truncate_string(phone_number, 20, "phone_number")
            name = self._truncate_string(name, 100, "name") if name else None
            location = self._truncate_string(location, 100, "location") if location else None
            request_details = self._truncate_string(request_details, 1000, "request_details") if request_details else None
            
            # Insert or update kit_requests table
            self.supabase_client.table("kit_requests").upsert({
                "phone_number": phone_number,
                "name": name,
                "location": location,
                "request_details": request_details,
                "created_at": datetime.now().isoformat()
            }).execute()
            logger.info(f"Successfully added/updated kit request: phone_number={phone_number}, name={name}, location={location}")
        except Exception as e:
            logger.error(f"Error adding/updating kit request for phone_number={phone_number}: {e}", exc_info=True)
            raise

    def get_test_result(self, phone_number: str, message: str = None) -> str:
        """
        Retrieve test result for a phone number from the AFYA KE system.

        Args:
            phone_number (str): User's phone number
            message (str, optional): User's message (for extracting AFYA KE ID if needed)

        Returns:
            str: Test result or None if not found
        """
        try:
            if not self._validate_phone_number(phone_number):
                raise ValueError("Invalid phone number for test result retrieval")
            
            phone_number = self._truncate_string(phone_number, 20, "phone_number")
            
            # Query test_results table (placeholder for AFYA KE integration)
            response = self.supabase_client.table("test_results").select("result").eq("phone_number", phone_number).limit(1).execute()
            if response.data:
                return response.data[0]["result"]
            logger.info(f"No test result found for phone_number={phone_number}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving test result for phone_number={phone_number}: {e}", exc_info=True)
            return None

    def get_kit_request_count(self) -> int:
        """
        Retrieve the total number of kit requests for analytics.

        Returns:
            int: Number of kit requests
        """
        try:
            response = self.supabase_client.table("kit_requests").select("count", count="exact").execute()
            return response.data[0]["count"] if response.data else 0
        except Exception as e:
            logger.error(f"Error retrieving kit request count: {e}", exc_info=True)
            return 0

    def get_screening_completion_count(self) -> int:
        """
        Retrieve the total number of completed screenings (test results) for analytics.

        Returns:
            int: Number of completed screenings
        """
        try:
            response = self.supabase_client.table("test_results").select("count", count="exact").execute()
            return response.data[0]["count"] if response.data else 0
        except Exception as e:
            logger.error(f"Error retrieving screening completion count: {e}", exc_info=True)
            return 0

    def close(self):
        """Clean up Supabase client (no-op as supabase-py manages connections)."""
        logger.info("Supabase client does not require explicit cleanup")

    def __del__(self):
        """Clean up on object deletion (no-op for Supabase client)."""
        pass