import json
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from config import Config
from handlers.webhook_handler import WebhookHandler
from handlers.greeting_handler import GreetingHandler
from handlers.ai_handler import AIHandler
from utils.session_manager import SessionManager
from utils.data_manager import DataManager
from services.whatsapp_service import WhatsAppService
from message_processor import MessageProcessor

# Load environment variables from .env file
load_dotenv()

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("afyabot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__)

# Initialize the configuration object
config = Config()

# --- Initialize core service objects ---
try:
    session_manager = SessionManager(config.SESSION_TIMEOUT)
    data_manager = DataManager(config)
    whatsapp_service = WhatsAppService(config)
    
    logger.info("Core services initialized for AfyaBot (Gynecology Medical AI)")
    
except Exception as e:
    logger.error(f"Error initializing core services: {e}", exc_info=True)
    exit(1)

# --- Initialize handlers for AfyaBot ---
try:
    greeting_handler = GreetingHandler(config, session_manager, data_manager, whatsapp_service)
    ai_handler = AIHandler(config, session_manager, data_manager, whatsapp_service)
    
    # Initialize message processor
    message_processor = MessageProcessor(
        config, 
        session_manager, 
        data_manager, 
        whatsapp_service
    )
    
    logger.info("AfyaBot handlers and message processor initialized.")
    
except Exception as e:
    logger.error(f"Error initializing handlers: {e}", exc_info=True)
    exit(1)

# Initialize the WebhookHandler with message processor
try:
    webhook_handler = WebhookHandler(config, message_processor)
    logger.info("WebhookHandler initialized with message processor.")
except Exception as e:
    logger.error(f"Error initializing WebhookHandler: {e}", exc_info=True)
    exit(1)

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """
    Endpoint for WhatsApp webhook verification.
    Handles the GET request from Meta to verify the webhook URL.
    """
    return webhook_handler.verify_webhook(request)

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Endpoint for receiving incoming WhatsApp messages.
    Handles the POST request containing message data from WhatsApp.
    """
    return webhook_handler.handle_webhook(request)

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint for monitoring service status.
    """
    try:
        # Basic health checks
        ai_status = "enabled" if ai_handler.ai_enabled else "disabled"
        
        return jsonify({
            "status": "healthy",
            "service": "AfyaBot - Gynecology Medical AI",
            "ai_service": ai_status,
            "session_count": len(session_manager._sessions) if hasattr(session_manager, '_sessions') else 0
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """
    Endpoint to get basic analytics about conversations and usage.
    """
    try:
        # Basic analytics
        total_sessions = len(session_manager._sessions) if hasattr(session_manager, '_sessions') else 0
        # Placeholder: Add medical-specific metrics if supported by DataManager
        kit_requests = data_manager.get_kit_request_count() if hasattr(data_manager, 'get_kit_request_count') else 0
        screening_completions = data_manager.get_screening_completion_count() if hasattr(data_manager, 'get_screening_completion_count') else 0
        
        return jsonify({
            "status": "success",
            "data": {
                "total_active_sessions": total_sessions,
                "kit_requests": kit_requests,
                "screening_completions": screening_completions,
                "ai_service_status": "enabled" if ai_handler.ai_enabled else "disabled"
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/cleanup", methods=["POST"])
def manual_cleanup():
    """
    Endpoint to manually trigger cleanup of expired sessions.
    """
    try:
        message_processor.cleanup_expired_resources()
        return jsonify({
            "status": "success",
            "message": "Cleanup completed successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error during manual cleanup: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({"status": "error", "message": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting AfyaBot - Gynecology Medical AI WhatsApp Bot...")
    logger.info(f"Webhook URL: {config.CALLBACK_BASE_URL}/webhook")
    logger.info(f"Health Check: {config.CALLBACK_BASE_URL}/health")
    logger.info(f"Analytics: {config.CALLBACK_BASE_URL}/api/analytics")
    logger.info("Logs: Check afyabot.log file for detailed logs")
    logger.info(f"AI Service Status: {'Enabled' if ai_handler.ai_enabled else 'Disabled'}")
    logger.info("To run this application, use Gunicorn from your terminal:")
    logger.info(
        "gunicorn -w 4 -k gevent --timeout 120 --preload -b 0.0.0.0:{port} app:app".format(
            port=config.APP_PORT
        )
    )
    logger.info("AfyaBot is ready to assist with women's health queries! 🌸")
    
    # Run the Flask development server (for development only)
    app.run(debug=True, host="0.0.0.0", port=config.APP_PORT)