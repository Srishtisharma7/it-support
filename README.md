# AI IT Support Chatbot

A small prototype of an AI-powered IT support bot.

- **FastAPI** backend with an async `/chat` endpoint
- **LLM tool calling**: the model decides when to call `create_ticket` or `get_ticket`
- **Conversation memory**: history is kept per `session_id`
- **Mock ticketing** (stand-in for ServiceNow), saved in `tickets.json`
- Error handling, logging and tests (the tests use a fake LLM, no API key needed).

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your free API key (e.g. from console.groq.com)
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs and try `POST /chat`:

```json
{"message": "My VPN is not connecting and I have a meeting in 10 minutes"}
```

Reuse the returned `session_id` in the next request to continue the same conversation.

## Test

```bash
pytest
```

## Next steps
- Add a knowledge base with retrieval (RAG)
- Replace the mock tickets with a real ServiceNow API
- Move session memory to a database
