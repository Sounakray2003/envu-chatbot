#!/usr/bin/env python3
"""Simple manual test for ConversationService.
Run this script after installing dependencies and setting OpenAI env vars.
It creates a sample request payload, invokes the service, and prints the response.
"""

import os
import json
from pathlib import Path

# Ensure project root is in PYTHONPATH
project_root = Path(__file__).resolve().parent
os.chdir(project_root)

# Import the service (adjust import path if needed)
from services.conversation_service import ConversationService

def main():
    # Sample request payload – adapt to your knowledge base / collection
    request = {
        "user_id": "user_123",
        "session_id": None,  # let the service generate one
        "question": "Tell me about Infosys",
        "collection": "main_memory",
        "history": []  # optional previous turns
    }
    service = ConversationService()
    response = service.handle_question(request)
    print("--- Response ---")
    print(json.dumps(response, indent=2, ensure_ascii=True))

if __name__ == "__main__":
    main()
