#!/usr/bin/env python3
"""
Wallpaper.ng AI Assistant Runner Script

This script provides a convenient way to run the Wallpaper.ng AI assistant with different configurations
and includes startup checks to ensure everything is properly configured.
"""

import os
import sys
import logging
import json
from pathlib import Path

def check_environment():
    """Check if all required environment variables and files are present."""
    print("🔍 Checking environment setup...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ ERROR: .env file not found!")
        print("   Please create a .env file with your configuration.")
        print("   See the documentation for required environment variables.")
        return False
    
    # Check required environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        'WHATSAPP_ACCESS_TOKEN',
        'WHATSAPP_PHONE_NUMBER_ID', 
        'VERIFY_TOKEN'
    ]
    
    # Optional AI service variables
    ai_vars = [
        'AZURE_API_KEY',
        'AZURE_ENDPOINT',
        'AZURE_DEPLOYMENT_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ ERROR: Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("   Please add these to your .env file.")
        return False
    
    # Check AI service variables (optional but recommended)
    missing_ai_vars = []
    for var in ai_vars:
        if not os.getenv(var):
            missing_ai_vars.append(var)
    
    if missing_ai_vars:
        print(f"⚠️  WARNING: Missing AI service variables (AI features will be disabled):")
        for var in missing_ai_vars:
            print(f"   - {var}")
        print("   Add these to your .env file to enable AI-powered responses.")
    else:
        print("✅ AI service variables found - AI features will be enabled!")
    
    # Check directory structure
    required_dirs = ['handlers', 'services', 'utils']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"❌ ERROR: Directory '{directory}' not found!")
            print("   Please ensure the project structure is correct.")
            return False
        
        # Check for __init__.py files
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            print(f"⚠️  WARNING: {init_file} not found. Creating...")
            Path(init_file).touch()
    
    print("✅ Environment check completed successfully!")
    return True

def setup_logging(debug=False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("wallpapers_bot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from requests library
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def print_startup_info():
    """Print startup information and instructions."""
    print("\n" + "="*70)
    print("🎨 Wallpaper.ng AI Assistant")
    print("="*70)
    print("🤖 AI Assistant Features:")
    print("   • Conversational product information and pricing")
    print("   • Custom design consultation and recommendations")
    print("   • Installation advice and service information")
    print("   • Quote generation for projects")
    print("   • Nationwide delivery information")
    print("   • 24/7 automated customer support")
    print("\n📋 Next Steps:")
    print("   1. Make sure ngrok is running: ngrok http 8000")
    print("   2. Update your WhatsApp webhook URL in Meta Developer Console")
    print("   3. Test the assistant by sending a message to your WhatsApp number")
    print("   4. Try asking questions like:")
    print("      - 'What types of wallpapers do you have?'")
    print("      - 'How much does installation cost?'")
    print("      - 'Can you help me design my living room?'")
    print("\n🔗 Important URLs:")
    print("   • Webhook: http://localhost:8000/webhook")
    print("   • Health Check: http://localhost:8000/health")
    print("   • Analytics: http://localhost:8000/api/analytics")
    print("   • Logs: Check wallpapers_bot.log file for detailed logs")
    print("="*70 + "\n")

def check_ai_service():
    """Check if AI service is properly configured."""
    from dotenv import load_dotenv
    load_dotenv()
    
    ai_key = os.getenv('AZURE_API_KEY')
    ai_endpoint = os.getenv('AZURE_ENDPOINT')
    
    if ai_key and ai_endpoint:
        print("✅ AI Service: Enabled (Azure OpenAI configured)")
        return True
    else:
        print("⚠️  AI Service: Disabled (Missing Azure OpenAI configuration)")
        print("   The assistant will provide basic responses without AI features.")
        return False

def main():
    """Main function to run the Wallpaper.ng AI assistant."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wallpaper.ng AI Assistant')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--no-check', action='store_true', help='Skip environment checks')
    parser.add_argument('--production', action='store_true', help='Run in production mode (use with Gunicorn)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Check environment unless --no-check is specified
        if not args.no_check:
            if not check_environment():
                logger.error("Environment check failed. Please fix the issues above.")
                sys.exit(1)
            
            # Check AI service configuration
            check_ai_service()
        
        # Print startup information
        print_startup_info()
        
        # Production mode instructions
        if args.production:
            print("🚀 PRODUCTION MODE:")
            print("   This script is for development. For production, use:")
            print(f"   gunicorn -w 4 -k gevent --timeout 120 --preload -b 0.0.0.0:{args.port} app:app")
            print("\n   Or use a process manager like PM2 or systemd.")
            return
        
        # Import and run the Flask app
        logger.info("Starting Wallpaper.ng AI Assistant server...")
        
        from app import app
        
        # Run the Flask application
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False  # Disable reloader to avoid double startup messages
        )
        
    except KeyboardInterrupt:
        logger.info("AI Assistant server stopped by user.")
        print("\n👋 Wallpaper.ng AI Assistant stopped. Goodbye!")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("❌ ERROR: Could not import required modules.")
        print("   Make sure all files are in place and dependencies are installed.")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ ERROR: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()