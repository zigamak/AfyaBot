import logging
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DBManager:
    """Manages database interactions for BEDC WhatsApp Bot using PostgreSQL."""

    def __init__(self, config=None):
        """
        Initialize DBManager with PostgreSQL connection.

        Args:
            config: Configuration object or dictionary with database credentials
        """
        # Get database connection string
        try:
            if isinstance(config, dict):
                self.db_url = config.get("database_url") or os.getenv("DATABASE_URL")
            else:
                self.db_url = getattr(config, 'DATABASE_URL', os.getenv("DATABASE_URL"))
        except:
            self.db_url = os.getenv("DATABASE_URL")
        
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in config or environment variables")
        
        # Initialize connection pool
        try:
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.db_url
            )
            logger.info("Database connection pool initialized successfully")
            
            # Test connection
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    logger.info("Database connection test successful")
                    
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self.pool.putconn(conn)

    def get_customer_by_account(self, account_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve customer information by account number.

        Args:
            account_number (str): Customer's account number

        Returns:
            Optional[Dict[str, Any]]: Customer data or None if not found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            id,
                            account_number,
                            customer_name,
                            phone_number,
                            email,
                            address,
                            feeder,
                            meter_number,
                            is_metered AS metered,
                            bill_amount,
                            nerc_cap,
                            customer_category,
                            status,
                            created_at,
                            updated_at
                        FROM customers
                        WHERE account_number = %s
                        LIMIT 1
                    """, (account_number,))
                    
                    result = cur.fetchone()
                    
                    if result:
                        customer = dict(result)
                        logger.info(f"Found customer: {account_number}")
                        return customer
                    else:
                        logger.info(f"Customer not found: {account_number}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error retrieving customer {account_number}: {e}")
            return None

    def get_customer_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve customer information by phone number.

        Args:
            phone_number (str): Customer's phone number

        Returns:
            Optional[Dict[str, Any]]: Customer data or None if not found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            id,
                            account_number,
                            customer_name,
                            phone_number,
                            email,
                            address,
                            feeder,
                            meter_number,
                            is_metered AS metered,
                            bill_amount,
                            nerc_cap,
                            customer_category,
                            status
                        FROM customers
                        WHERE phone_number = %s
                        LIMIT 1
                    """, (phone_number,))
                    
                    result = cur.fetchone()
                    
                    if result:
                        logger.info(f"Found customer by phone: {phone_number}")
                        return dict(result)
                    else:
                        logger.info(f"Customer not found by phone: {phone_number}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error retrieving customer by phone {phone_number}: {e}")
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
            
            bill_amount = float(customer.get("bill_amount", 0))
            nerc_cap = float(customer.get("nerc_cap", 0))
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
                         assistant_response: str, intent: str = None, account_number: str = None):
        """
        Save a conversation entry.

        Args:
            phone_number (str): User's phone number
            session_id (str): Session identifier
            user_message (str): User's message
            assistant_response (str): Bot's response
            intent (str): Detected intent (optional)
            account_number (str): Account number if available (optional)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO conversations 
                        (session_id, phone_number, user_message, assistant_response, intent, account_number, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        session_id,
                        phone_number,
                        user_message,
                        assistant_response,
                        intent,
                        account_number,
                        datetime.now()
                    ))
                    
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
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            id,
                            session_id,
                            user_message,
                            assistant_response,
                            intent,
                            account_number,
                            created_at
                        FROM conversations
                        WHERE phone_number = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (phone_number, limit))
                    
                    results = cur.fetchall()
                    conversations = [dict(row) for row in results]
                    
                    # Reverse to get chronological order
                    conversations.reverse()
                    
                    return conversations
                    
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
            # Generate reference number
            reference_number = f"FR-{account_number}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO fault_reports 
                        (reference_number, phone_number, account_number, email, fault_description, 
                         fault_type, status, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        reference_number,
                        phone_number,
                        account_number,
                        email,
                        fault_description,
                        'power_outage',
                        'pending',
                        datetime.now(),
                        datetime.now()
                    ))
                    
            logger.info(f"Saved fault report {reference_number} for account {account_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving fault report: {e}")
            return False

    def get_fault_reports_by_account(self, account_number: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get fault reports for a specific account.

        Args:
            account_number (str): Customer's account number
            limit (int): Maximum number of reports to return

        Returns:
            List[Dict[str, Any]]: List of fault reports
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            id,
                            reference_number,
                            phone_number,
                            account_number,
                            email,
                            fault_description,
                            fault_type,
                            status,
                            priority,
                            assigned_to,
                            resolution_notes,
                            resolved_at,
                            created_at,
                            updated_at
                        FROM fault_reports
                        WHERE account_number = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (account_number, limit))
                    
                    results = cur.fetchall()
                    return [dict(row) for row in results]
                    
        except Exception as e:
            logger.error(f"Error retrieving fault reports: {e}")
            return []

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
            # Generate application number
            application_number = f"MAP-{datetime.now().strftime('%Y%m%d')}-{account_number}"
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO map_applications 
                        (application_number, phone_number, account_number, customer_name, email, 
                         status, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        application_number,
                        phone_number,
                        account_number,
                        customer_name,
                        email,
                        'pending',
                        datetime.now(),
                        datetime.now()
                    ))
                    
            logger.info(f"Saved MAP application {application_number} for account {account_number}")
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
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            id,
                            feeder_name,
                            feeder_code,
                            nerc_cap_residential,
                            nerc_cap_commercial,
                            nerc_cap_industrial,
                            location,
                            status
                        FROM feeders
                        WHERE feeder_name = %s
                        LIMIT 1
                    """, (feeder_name,))
                    
                    result = cur.fetchone()
                    
                    if result:
                        return dict(result)
                    else:
                        return None
                        
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
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get total customers
                    cur.execute("SELECT COUNT(*) as total FROM customers WHERE status = 'active'")
                    total_customers = cur.fetchone()['total']
                    
                    # Get total conversations
                    cur.execute("SELECT COUNT(*) as total FROM conversations")
                    total_conversations = cur.fetchone()['total']
                    
                    # Get total fault reports
                    cur.execute("SELECT COUNT(*) as total FROM fault_reports")
                    total_fault_reports = cur.fetchone()['total']
                    
                    # Get total MAP applications
                    cur.execute("SELECT COUNT(*) as total FROM map_applications")
                    total_map_applications = cur.fetchone()['total']
                    
                    # Count customers above NERC cap
                    cur.execute("""
                        SELECT COUNT(*) as total 
                        FROM customers 
                        WHERE status = 'active' AND bill_amount > nerc_cap
                    """)
                    above_cap_count = cur.fetchone()['total']
                    
                    # Count unmetered customers
                    cur.execute("""
                        SELECT COUNT(*) as total 
                        FROM customers 
                        WHERE status = 'active' AND is_metered = FALSE
                    """)
                    unmetered_customers = cur.fetchone()['total']
                    
                    return {
                        "total_customers": total_customers,
                        "total_conversations": total_conversations,
                        "total_fault_reports": total_fault_reports,
                        "total_map_applications": total_map_applications,
                        "customers_above_cap": above_cap_count,
                        "unmetered_customers": unmetered_customers
                    }
                    
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {}

    def save_faq_interaction(self, session_id: str, phone_number: str, 
                            question: str, answer: str, category: str = None) -> bool:
        """
        Save FAQ interaction for analytics.

        Args:
            session_id (str): Session identifier
            phone_number (str): User's phone number
            question (str): User's question
            answer (str): FAQ answer provided
            category (str): FAQ category (optional)

        Returns:
            bool: True if saved successfully
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO faq_interactions 
                        (session_id, phone_number, question, answer, category, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        session_id,
                        phone_number,
                        question,
                        answer,
                        category,
                        datetime.now()
                    ))
                    
            logger.info(f"Saved FAQ interaction for {phone_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving FAQ interaction: {e}")
            return False

    def update_customer_info(self, account_number: str, **kwargs) -> bool:
        """
        Update customer information.

        Args:
            account_number (str): Customer's account number
            **kwargs: Fields to update (email, phone_number, address, etc.)

        Returns:
            bool: True if updated successfully
        """
        try:
            # Build dynamic UPDATE query
            update_fields = []
            values = []
            
            allowed_fields = ['email', 'phone_number', 'address', 'customer_name', 
                            'bill_amount', 'nerc_cap', 'is_metered', 'meter_number']
            
            for field, value in kwargs.items():
                if field in allowed_fields:
                    update_fields.append(f"{field} = %s")
                    values.append(value)
            
            if not update_fields:
                logger.warning("No valid fields to update")
                return False
            
            # Add account_number to values
            values.append(account_number)
            
            query = f"""
                UPDATE customers 
                SET {', '.join(update_fields)}, updated_at = %s
                WHERE account_number = %s
            """
            values.insert(-1, datetime.now())
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, values)
                    
            logger.info(f"Updated customer info for account {account_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating customer info: {e}")
            return False

    def get_pending_fault_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get pending fault reports for admin dashboard.

        Args:
            limit (int): Maximum number of reports to return

        Returns:
            List[Dict[str, Any]]: List of pending fault reports
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            fr.*,
                            c.customer_name,
                            c.feeder,
                            c.address
                        FROM fault_reports fr
                        LEFT JOIN customers c ON fr.account_number = c.account_number
                        WHERE fr.status = 'pending'
                        ORDER BY fr.created_at DESC
                        LIMIT %s
                    """, (limit,))
                    
                    results = cur.fetchall()
                    return [dict(row) for row in results]
                    
        except Exception as e:
            logger.error(f"Error retrieving pending fault reports: {e}")
            return []

    def close(self):
        """Close all database connections."""
        try:
            if self.pool:
                self.pool.closeall()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")