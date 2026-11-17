"""AI IT Support Chatbot: FastAPI + LLM tool calling + conversation memory."""
import json
import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chatbot")

# Any OpenAI-compatible API works. Defaults are for Groq's free tier.
client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY", "missing-key"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
)
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_STEPS = 4  # max LLM <-> tool round trips per user message

SYSTEM_PROMPT = (
    "You are an IT support assistant. Help users with laptop, network, email and "
    "software problems. Ask a short clarifying question if the issue is unclear. "
    "If you cannot solve it in chat, create a ticket with the create_ticket tool. "
    "Use get_ticket when the user asks about an existing ticket. Keep replies short."
)

app = FastAPI(title="AI IT Support Chatbot")
sessions: dict[str, list[dict]] = {}  # session_id -> message history (memory)
TICKET_FILE = Path("tickets.json")  # mock ServiceNow


# ---------- Tools (mock ticketing system) ----------
def _load_tickets() -> dict:
    return json.loads(TICKET_FILE.read_text()) if TICKET_FILE.exists() else {}


def create_ticket(issue: str, priority: str = "medium") -> dict:
    tickets = _load_tickets()
    ticket_id = f"INC{len(tickets) + 1:04d}"
    tickets[ticket_id] = {"issue": issue, "priority": priority, "status": "open"}
    TICKET_FILE.write_text(json.dumps(tickets, indent=2))
    log.info("Created ticket %s", ticket_id)
    return {"ticket_id": ticket_id, "status": "open"}


def get_ticket(ticket_id: str) -> dict:
    ticket = _load_tickets().get(ticket_id)
    return ticket or {"error": f"Ticket {ticket_id} not found"}


TOOL_FUNCTIONS = {"create_ticket": create_ticket, "get_ticket": get_ticket}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create an IT support ticket when the issue cannot be solved in chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string", "description": "Short summary of the problem"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["issue"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": "Look up an existing ticket by its ID, e.g. INC0001.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
        },
    },
]


def run_tool(name: str, args: dict) -> dict:
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool {name}"}
    try:
        return func(**args)
    except Exception as exc:  # tool failures should not crash the chat
        log.exception("Tool %s failed", name)
        return {"error": str(exc)}


# ---------- API ----------
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.setdefault(session_id, [{"role": "system", "content": SYSTEM_PROMPT}])
    history.append({"role": "user", "content": req.message})

    try:
        for _ in range(MAX_STEPS):
            resp = await client.chat.completions.create(model=MODEL, messages=history, tools=TOOLS)
            msg = resp.choices[0].message

            if not msg.tool_calls:  # final answer
                history.append({"role": "assistant", "content": msg.content})
                return ChatResponse(session_id=session_id, reply=msg.content or "")

            history.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in msg.tool_calls
                ],
            })
            for call in msg.tool_calls:
                result = run_tool(call.function.name, json.loads(call.function.arguments or "{}"))
                log.info("Tool %s -> %s", call.function.name, result)
                history.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    except Exception as exc:
        log.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    raise HTTPException(status_code=500, detail="Too many tool steps")


@app.get("/tickets")
async def list_tickets():
    return _load_tickets()


@app.get("/health")
async def health():
    return {"status": "ok"}
