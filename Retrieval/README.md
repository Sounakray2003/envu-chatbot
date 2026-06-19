# 📚 Conversational RAG Chatbot

**Production‑grade multi‑turn Retrieval‑Augmented Generation** built with OpenAI, Qdrant, and a lightweight in‑memory session manager.

---

## 🎯 Goal
Transform a single‑turn RAG prototype into a robust conversational system that:

- Handles arbitrary session IDs (auto‑generated when omitted)
- Persists turn history for context‑aware follow‑ups
- Rewrites follow‑up questions into standalone queries using the OpenAI chat model
- Retrieves relevant chunks from a Qdrant vector store
- Returns a concise answer together with the raw retrieved chunks

---

## 📂 Project Structure
```
q0_knowledge_base/
│
├─ services/                     # Core orchestration layer
│   ├─ conversation_service.py   # ★ New – main entry point
│   ├─ query_rewriter.py         # LLM‑based query rewriting
│   └─ session_manager.py        # Thread‑safe in‑memory session store
│
├─ retrieval_new.py              # Existing dense‑search implementation (unchanged)
├─ test_conversation_service.py  # Simple manual test script
├─ requirements.txt               # Pin‑compatible dependencies (see below)
└─ README.md                     # 📖 This file
```

---

## 🛠️ Setup – One‑Time
> **All commands are meant to be run from the project root**
> `C:\Users\kruti\Downloads\chatbot\q0_knowledge_base`

```powershell
# 1️⃣ Create an isolated virtual environment
python -m venv .venv

# 2️⃣ Activate it (PowerShell)
.\.venv\Scripts\Activate.ps1

# 3️⃣ Verify activation (you should see the .venv prefix in the prompt)
python --version   # → should point to .venv\Scripts\python.exe

# 4️⃣ Install the pinned, conflict‑free stack
pip install -r requirements.txt

# 5️⃣ (Optional) Freeze exact versions for CI / production
pip freeze > locked_requirements.txt
```

---

## 🚀 Quick Start – Run the Demo
```powershell
# With the venv still active, run the provided test script
python test_conversation_service.py
```
You will see a JSON payload printed to the console, e.g.:
```json
{
  "answer": "Tell me about Envu",
  "session_id": "auto:7f6c2a0e6b0c8b3c4d5e6f7a8b9c0d1e",
  "retrieved_chunks": [
    {"text": "Envu is environmental science company …", "score": 0.94, "metadata": {"doc_id": "12345"}},
    ...
  ]
}
```
The `session_id` can be reused for subsequent calls to maintain context.

---

## 📡 Using the Service in Your Code
```python
from services.conversation_service import ConversationService

svc = ConversationService()
payload = {
    "user_id": "alice",
    "session_id": None,          # let the service generate one
    "question": "What does the chatbot do?",
    "history": []               # optional – ignored, history is pulled from SessionManager
}
response = svc.handle_question(payload)
print(response["answer"])
```
The service automatically:
1. Generates (or re‑uses) a session ID.
2. Pulls the last 10 turns from the session store.
3. Calls `QueryRewriter` to turn follow‑up queries into standalone text.
4. Retrieves relevant vectors via `retrieve()`.
5. Returns a friendly answer and the raw chunks.

---

## 🧩 Architecture Overview
| Component | Responsibility |
|-----------|----------------|
| **ConversationService** (new) | Orchestrates the whole flow – session handling, query rewriting, retrieval, answer formatting, and persisting the turn. |
| **SessionManager** | Thread‑safe in‑memory dictionary (`{session_id: [messages...]}`) with `add_message`/`get_history`. |
| **QueryRewriter** | Calls OpenAI's `gpt‑4o‑mini` (or any Chat model) with the conversation history to produce a *standalone* query. |
| **retrieval_new.retrieve** | Existing dense‑search pipeline that embeds the query, hits Qdrant, and returns `RetrievalResult` objects. |
| **Qdrant** | Vector database storing your knowledge‑base embeddings. |
| **FastAPI (optional)** | You can expose `ConversationService.handle_question` as a `/chat` endpoint – just import the service and return the dict. |

---

## 🧪 Testing & Validation
1. **Unit‑style test** – the `test_conversation_service.py` script is a minimal sanity check.
2. **Multi‑turn flow** – in a Python REPL run:
   ```python
   svc = ConversationService()
   r1 = svc.handle_question({"user_id":"bob","question":"Tell me about Infosys"})
   r2 = svc.handle_question({"user_id":"bob","session_id":r1["session_id"],"question":"When was it founded?"})
   print(r2["answer"])   # should reference Infosys context from r1
   ```
3. **API test** – if you expose via FastAPI, start the server (`uvicorn main:app --reload`) and `POST /chat` with the same JSON schema.

---

## 📦 Dependency Pinning (requirements.txt)
```text
openai>=1.0.0
qdrant-client>=1.9.0
tiktoken>=0.6.0

# LangChain ecosystem (compatible 0.2.x series)
langchain==0.2.5
langchain-core==0.2.9
langchain-community==0.2.5
langchain-chroma==0.2.2
langchain-ollama==0.3.0
langchain-openai==0.1.14
langchain-text-splitters==0.2.1
langsmith==0.1.27

python-dotenv==1.0.1
numpy==1.26.4
opentelemetry-api==1.25.0
opentelemetry-sdk==1.25.0
opentelemetry-exporter-otlp-proto-common==1.25.0
opentelemetry-exporter-otlp-proto-http==1.25.0

# Your existing heavy‑weight libs (keep the versions you like)
transformers==4.48.3
sentence-transformers==3.0.1
pydantic==2.12.5
fastapi==0.115.12
uvicorn==0.34.2
```
Feel free to edit the file if you add new dependencies.

---

## ⚡️ Next Steps / Ideas
- Swap the in‑memory `SessionManager` for a persistent store (Redis, SQLite) for production.
- Replace the simple concatenation in `_format_answer` with a full LLM generation prompt (Template B in the design doc).
- Add unit tests (pytest) for `QueryRewriter` and `ConversationService`.
- Deploy the FastAPI app behind a reverse proxy (NGINX) and enable HTTPS.

---

## 🙋‍♀️ Need Help?
Open an issue, or drop a message in the repo’s Discussions. Happy building! 🌟

---

## Gupshup WhatsApp Integration

The FastAPI app now exposes a Gupshup webhook at `POST /gupshup/whatsapp/webhook`.

Set these environment variables before running the service:

```text
GUPSHUP_ROUTE_ENABLED=true
GUPSHUP_DEFAULT_COLLECTION=main_memory
GUPSHUP_DEFAULT_LIMIT=5
GUPSHUP_API_KEY=your_gupshup_api_key
GUPSHUP_APP_NAME=your_gupshup_app_name
GUPSHUP_SOURCE=your_whatsapp_source_number
GUPSHUP_SEND_MESSAGE_URL=https://api.gupshup.io/wa/api/v1/msg
GUPSHUP_HISTORY_STORE_PATH=./data/gupshup_history.json
GUPSHUP_HISTORY_STORE_LIMIT=10
```

The optional history store keeps WhatsApp sessions across restarts.

Configure the Gupshup callback/webhook URL to point at:

```text
https://your-domain.example/gupshup/whatsapp/webhook
```

Inbound text messages are mapped to a `session_id` using the sender's phone number, and the service acknowledges the webhook quickly before sending the answer back through Gupshup's outbound messaging API.

## Test the Running Retrieval API
If the Docker container is already up, confirm the API is healthy:

```powershell
Invoke-RestMethod -Uri http://localhost:8093/retrieve/health
```

Then send a retrieval request from PowerShell:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:8093/retrieve `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query":"Tell me about Infosys","session_id":"demo-session","collection":"main_memory"}'
```

If your terminal prompt looks like `C:\...>` without `PS`, you are in Command Prompt, not PowerShell. Use this `curl` command there instead:

```cmd
curl -X POST http://localhost:8093/retrieve -H "Content-Type: application/json" -d "{\"query\":\"Tell me about Infosys\",\"session_id\":\"demo-session\",\"collection\":\"main_memory\"}"
```

The retrieval API runs at `http://localhost:8093`; opening that URL directly in a browser will not run a query because `/retrieve` expects a POST request.


## Operational Endpoints

- `GET /health` ? service status
- `GET /gupshup/health` ? Gupshup WhatsApp channel status and session-store info
