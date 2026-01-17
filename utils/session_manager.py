import datetime
import logging
from threading import Lock

logger = logging.getLogger(__name__)

# Global in-memory store for sessions, protected by a Lock for thread safety.
_sessions_store = {}
_sessions_lock = Lock()

class SessionManager:
    """Manages user sessions and their states."""

    # Default timeouts for different session types
    SESSION_TIMEOUT_SECONDS = 3000  # 50 minutes for sessions
    PAID_SESSION_TIMEOUT_SECONDS = 12000

    # Short grace period for freshly reset sessions
    FRESH_RESET_GRACE_PERIOD_SECONDS = 2

    def __init__(self, session_timeout=None):
        self.timeout_minutes = session_timeout / 60 if session_timeout else self.SESSION_TIMEOUT_SECONDS / 60
        logger.info("SessionManager initialized. Default timeouts: Unpaid=%ds, Paid=%ds",
                     self.SESSION_TIMEOUT_SECONDS, self.PAID_SESSION_TIMEOUT_SECONDS)

    def _get_timeout_duration(self, session_data: dict) -> int:
        """Determines the correct timeout duration based on whether the session is marked as paid."""
        if session_data.get('is_paid_user') and session_data.get('extended_session'):
            paid_expires_str = session_data.get("paid_session_expires")
            if paid_expires_str:
                try:
                    paid_expires = datetime.datetime.fromisoformat(paid_expires_str)
                    if datetime.datetime.now() < paid_expires:
                        return self.PAID_SESSION_TIMEOUT_SECONDS
                except ValueError:
                    logger.warning(f"Invalid 'paid_session_expires' format for session. Treating as unpaid.")
            else:
                logger.warning(f"Session marked as paid but missing 'paid_session_expires'. Treating as unpaid.")
        return self.SESSION_TIMEOUT_SECONDS

    def get_session_state(self, session_id: str) -> dict:
        """
        Retrieves the current session state for a given session ID.
        If the session does not exist or has timed out, it initializes a new one.
        Updates the 'last_activity' timestamp for active sessions.
        """
        with _sessions_lock:
            session_data = _sessions_store.get(session_id)

            if session_data:
                # Check for explicit paid session expiration first
                if session_data.get("is_paid_user") and session_data.get("extended_session"):
                    paid_expires_str = session_data.get("paid_session_expires")
                    if paid_expires_str:
                        try:
                            paid_expires = datetime.datetime.fromisoformat(paid_expires_str)
                            if datetime.datetime.now() > paid_expires:
                                logger.info(f"Paid session {session_id} expired. Resetting to normal session.")
                                self._reset_paid_session_internal(session_id, session_data)
                                session_data = _sessions_store.get(session_id)
                        except ValueError:
                            logger.warning(f"Invalid 'paid_session_expires' format for session {session_id} during retrieval. Resetting paid status.")
                            self._reset_paid_session_internal(session_id, session_data)
                            session_data = _sessions_store.get(session_id)

                # Now apply general timeout logic
                time_since_last_activity = (datetime.datetime.now() - session_data["last_activity"]).total_seconds()
                timeout_duration = self._get_timeout_duration(session_data)

                if time_since_last_activity > timeout_duration:
                    logger.info(f"Session {session_id} timed out after {time_since_last_activity:.2f} seconds (timeout limit: {timeout_duration}s). Resetting.")
                    
                    # Reset session, preserving user info (name, address, phone number, ACCOUNT NUMBER)
                    user_name = session_data.get("user_name")
                    address = session_data.get("address")
                    account_number = session_data.get("account_number")  # PRESERVE ACCOUNT NUMBER

                    # Create a new default state, preserving key user info
                    new_session_data = {
                        "current_state": "start",
                        "current_handler": "greeting_handler",
                        "cart": {},
                        "selected_category": None,
                        "selected_item": None,
                        "user_name": user_name,
                        "phone_number": session_id,
                        "address": address,
                        "account_number": account_number,  # RESTORE ACCOUNT NUMBER
                        "quantity_prompt_sent": False,
                        "last_activity": datetime.datetime.now(),
                        "payment_reference": None,
                        "order_id": None,
                        "total_cost": 0,
                        "is_paid_user": False,
                        "extended_session": False,
                        "recent_order_id": None,
                        "paid_session_expires": None,
                        "freshly_reset_timestamp": datetime.datetime.now(),
                        "conversation_history": [],
                        "fault_data": {},
                        "billing_checked": False,
                        "faq_category": None
                    }
                    _sessions_store[session_id] = new_session_data
                    return new_session_data
                else:
                    # Session is active and not timed out, update last activity
                    session_data["last_activity"] = datetime.datetime.now()
                    session_data["freshly_reset_timestamp"] = None
                    logger.debug(f"Session {session_id} retrieved (active). Activity updated.")
                    return session_data
            else:
                # Session does not exist, initialize a brand new one
                new_session_data = {
                    "current_state": "start",
                    "current_handler": "greeting_handler",
                    "cart": {},
                    "selected_category": None,
                    "selected_item": None,
                    "user_name": None,
                    "phone_number": session_id,
                    "address": None,
                    "account_number": None,  # INCLUDE ACCOUNT NUMBER
                    "quantity_prompt_sent": False,
                    "last_activity": datetime.datetime.now(),
                    "payment_reference": None,
                    "order_id": None,
                    "total_cost": 0,
                    "is_paid_user": False,
                    "extended_session": False,
                    "recent_order_id": None,
                    "paid_session_expires": None,
                    "freshly_reset_timestamp": None,
                    "conversation_history": [],
                    "fault_data": {},
                    "billing_checked": False,
                    "faq_category": None
                }
                _sessions_store[session_id] = new_session_data
                logger.info(f"New session {session_id} initialized.")
                return new_session_data

    def update_session_state(self, session_id: str, new_state_data: dict):
        """
        Explicitly updates the entire session state for a given session ID.
        """
        if not isinstance(new_state_data, dict):
            logger.error(f"Attempted to update session {session_id} with non-dictionary data (type: {type(new_state_data)}). Update aborted.")
            return

        with _sessions_lock:
            old_state_data = _sessions_store.get(session_id, {})

            # Ensure 'last_activity' is always updated on state persist
            new_state_data['last_activity'] = datetime.datetime.now()
            
            old_handler = old_state_data.get("current_handler")
            old_state = old_state_data.get("current_state")
            new_handler = new_state_data.get("current_handler")
            new_current_state = new_state_data.get("current_state")

            is_transitioning_to_greeting_state = (
                new_handler == "greeting_handler" and
                new_current_state in ["start", "greeting"]
            )
            was_already_in_greeting_state = (
                old_handler == "greeting_handler" and
                old_state in ["start", "greeting"]
            )

            if is_transitioning_to_greeting_state and not was_already_in_greeting_state:
                new_state_data["freshly_reset_timestamp"] = datetime.datetime.now()
                logger.debug(f"Session {session_id}: Setting freshly_reset_timestamp due to state transition to '{new_handler}'/'{new_current_state}'.")
            else:
                new_state_data["freshly_reset_timestamp"] = None
                
            _sessions_store[session_id] = new_state_data
            logger.debug(f"Session {session_id} state updated to '{new_state_data.get('current_state', 'N/A')}'")

    def update_session_activity(self, session_id: str):
        """Updates the 'last_activity' timestamp for a given session."""
        with _sessions_lock:
            if session_id in _sessions_store:
                _sessions_store[session_id]["last_activity"] = datetime.datetime.now()
                _sessions_store[session_id]["freshly_reset_timestamp"] = None
                logger.debug(f"Updated activity for session {session_id}")
            else:
                logger.warning(f"Attempted to update activity for non-existent session {session_id}.")

    def set_session_paid_status(self, session_id: str, paid_status: bool):
        """Explicitly sets the paid status for a session."""
        with _sessions_lock:
            session_data = _sessions_store.get(session_id)
            if session_data:
                session_data['is_paid_user'] = paid_status
                if paid_status:
                    session_data['extended_session'] = True
                    session_data['paid_session_expires'] = (
                        datetime.datetime.now() + datetime.timedelta(seconds=self.PAID_SESSION_TIMEOUT_SECONDS)
                    ).isoformat()
                    logger.info(f"Session {session_id} paid status set to {paid_status} and extended for {self.PAID_SESSION_TIMEOUT_SECONDS / 3600} hours.")
                else:
                    session_data['extended_session'] = False
                    session_data['recent_order_id'] = None
                    session_data['paid_session_expires'] = None
                    logger.info(f"Session {session_id} paid status set to {paid_status} (normal timeout).")

                session_data['last_activity'] = datetime.datetime.now()
                session_data['freshly_reset_timestamp'] = None
                self.update_session_state(session_id, session_data)
            else:
                logger.warning(f"Attempted to set paid status for non-existent session {session_id}.")

    def extend_session_for_paid_user(self, session_id: str, order_id: str, hours: int = 24):
        """Extend session timeout for paid users to allow order tracking."""
        try:
            with _sessions_lock:
                state = _sessions_store.get(session_id)
                if not state:
                    state = self.get_session_state(session_id)
                    logger.warning(f"Session {session_id} did not exist when extending for paid user; a new one was initialized.")

                state["is_paid_user"] = True
                state["extended_session"] = True
                state["recent_order_id"] = order_id
                
                paid_expires = datetime.datetime.now() + datetime.timedelta(hours=hours)
                state["paid_session_expires"] = paid_expires.isoformat()
                
                state["last_activity"] = datetime.datetime.now()
                state["freshly_reset_timestamp"] = None
                
                self.update_session_state(session_id, state)
                logger.info(f"Extended session for paid user {session_id} for {hours} hours. Order: {order_id}")
                
        except Exception as e:
            logger.error(f"Error extending session for paid user {session_id}: {e}", exc_info=True)

    def is_paid_user_session(self, session_id: str) -> bool:
        """Check if this is an active paid user session."""
        try:
            with _sessions_lock:
                state = _sessions_store.get(session_id)
                if not state:
                    return False

                if not state.get("is_paid_user") or not state.get("extended_session"):
                    return False
                
                paid_expires_str = state.get("paid_session_expires")
                if paid_expires_str:
                    try:
                        paid_expires = datetime.datetime.fromisoformat(paid_expires_str)
                        if datetime.datetime.now() > paid_expires:
                            logger.info(f"Paid session {session_id} expired during is_paid_user_session check. Resetting.")
                            self._reset_paid_session_internal(session_id, state)
                            return False
                    except ValueError:
                        logger.warning(f"Invalid 'paid_session_expires' format for session {session_id}. Resetting paid status.")
                        self._reset_paid_session_internal(session_id, state)
                        return False
                else:
                    logger.warning(f"Session {session_id} has 'extended_session' but no 'paid_session_expires'. Resetting.")
                    self._reset_paid_session_internal(session_id, state)
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error checking paid user session {session_id}: {e}", exc_info=True)
            return False

    def _reset_paid_session_internal(self, session_id: str, state: dict):
        """Internal helper to reset a paid session back to normal."""
        paid_keys = ["is_paid_user", "extended_session", "recent_order_id", "paid_session_expires"]
        for key in paid_keys:
            if key in state:
                del state[key]
        
        state["last_activity"] = datetime.datetime.now()
        state["freshly_reset_timestamp"] = None

        self.update_session_state(session_id, state)
        logger.info(f"Reset paid session for {session_id} back to normal session")
            
    def clear_session_cart(self, session_id: str):
        """Clear the cart for a specific session."""
        with _sessions_lock:
            if session_id in _sessions_store:
                _sessions_store[session_id]["cart"] = {}
                self.update_session_state(session_id, _sessions_store[session_id])
                logger.info(f"Cart cleared for session {session_id}")
            else:
                logger.warning(f"Attempted to clear cart for non-existent session {session_id}.")

    def reset_session_order_data(self, session_id: str):
        """Reset order-specific data in session."""
        with _sessions_lock:
            if session_id in _sessions_store:
                state = _sessions_store[session_id]
                state["order_id"] = None
                state["payment_reference"] = None
                state["total_cost"] = 0
                state["freshly_reset_timestamp"] = None
                self.update_session_state(session_id, state)
            else:
                logger.warning(f"Attempted to reset order data for non-existent session {session_id}.")

    def clear_full_session(self, session_id: str):
        """Completely removes a session from the manager's store."""
        with _sessions_lock:
            if session_id in _sessions_store:
                del _sessions_store[session_id]
                logger.info(f"Full session {session_id} cleared.")
            else:
                logger.warning(f"Attempted to clear non-existent session {session_id}.")

    def cleanup_expired_sessions(self):
        """Iterates through all sessions and removes those that have timed out."""
        cleaned_count = 0
        sessions_to_clear = []

        with _sessions_lock:
            for session_id, session_data in list(_sessions_store.items()):
                if session_data.get("is_paid_user") and session_data.get("extended_session"):
                    paid_expires_str = session_data.get("paid_session_expires")
                    if paid_expires_str:
                        try:
                            paid_expires = datetime.datetime.fromisoformat(paid_expires_str)
                            if datetime.datetime.now() > paid_expires:
                                logger.info(f"Cleanup: Paid session {session_id} explicitly expired. Resetting to normal.")
                                self._reset_paid_session_internal(session_id, session_data)
                        except ValueError:
                            logger.warning(f"Cleanup: Invalid 'paid_session_expires' format for session {session_id}. Resetting paid status.")
                            self._reset_paid_session_internal(session_id, session_data)
                    else:
                        logger.warning(f"Cleanup: Session {session_id} marked as paid but missing 'paid_session_expires'. Resetting.")
                        self._reset_paid_session_internal(session_id, session_data)

                time_since_last_activity = (datetime.datetime.now() - session_data["last_activity"]).total_seconds()
                timeout_duration = self._get_timeout_duration(session_data)
                
                if time_since_last_activity > timeout_duration:
                    sessions_to_clear.append(session_id)

            for session_id in sessions_to_clear:
                del _sessions_store[session_id]
                cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleanup job: Removed {cleaned_count} expired sessions.")
            return cleaned_count

    def is_freshly_reset(self, session_id: str) -> bool:
        """Checks if the session was recently reset to the greeting/start state."""
        with _sessions_lock:
            state = _sessions_store.get(session_id)
            if not state:
                return False

            freshly_reset_timestamp = state.get("freshly_reset_timestamp")
            current_handler = state.get("current_handler")
            current_state = state.get("current_state")

            if freshly_reset_timestamp and \
               (current_handler == "greeting_handler" and current_state in ["greeting", "start"]):
                time_since_reset = (datetime.datetime.now() - freshly_reset_timestamp).total_seconds()
                return time_since_reset < self.FRESH_RESET_GRACE_PERIOD_SECONDS
            return False

    def reset_freshly_reset_flag(self, session_id: str):
        """Manually resets the freshly_reset_timestamp for a session."""
        with _sessions_lock:
            if session_id in _sessions_store:
                _sessions_store[session_id]["freshly_reset_timestamp"] = None
                logger.debug(f"Freshly reset flag cleared for session {session_id}.")
            else:
                logger.warning(f"Attempted to clear freshly reset flag for non-existent session {session_id}.")