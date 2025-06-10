#!/bin/bash

# This script automates the startup of the Simple Chatbot server.

# Navigate to the server directory
cd "$(dirname "$0")/server" || exit

echo "-------------------------------------------------------------------"
echo " Simple Chatbot Server Runner"
echo "-------------------------------------------------------------------"
echo ""
echo "Important Prerequisites:"
echo ""
echo "1. Virtual Environment:"
echo "   - Ensure you have a Python virtual environment set up and activated."
echo "   - If not, you can create one in the 'examples/simple-chatbot/server/' directory by running:"
echo "     # python3 -m venv venv"
echo "   - And activate it (from the 'server' directory) using:"
echo "     # source venv/bin/activate  (on Linux/macOS)"
echo "     # venv\\Scripts\\activate    (on Windows)"
echo ""
echo "2. Dependencies:"
echo "   - Make sure all Python dependencies are installed from requirements.txt."
echo "   - If not, run (with your virtual environment activated):"
echo "     # pip install -r requirements.txt"
echo ""
echo "3. Environment Variables (.env file):"
echo "   - Ensure you have a .env file in the 'examples/simple-chatbot/server/' directory."
echo "   - Copy 'env.example' to '.env' if it doesn't exist."
echo "   - Configure your .env file with the necessary API keys:"
echo "     - DAILY_API_KEY (required)"
echo "     - GEMINI_API_KEY (required for this demo as BOT_IMPLEMENTATION should be 'gemini')"
echo "     - OPENAI_API_KEY (if you were to use 'openai' bot)"
echo "     - ELEVENLABS_API_KEY (if using ElevenLabs for TTS, though not default for Gemini bot)"
echo "   - Set BOT_IMPLEMENTATION in your .env file. For the elder demo, it should be:"
echo "     BOT_IMPLEMENTATION=gemini"
echo ""
echo "-------------------------------------------------------------------"
echo "Attempting to start the server..."
echo "Press Ctrl+C to stop the server."
echo "-------------------------------------------------------------------"
echo ""

# Run the server
# You can modify the host and port if needed, e.g., python server.py --host 127.0.0.1 --port 8000
python server.py

echo "Server stopped."
