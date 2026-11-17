"""Tests with a fake LLM, so no API key is needed. Run: pytest"""
import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

import main


class FakeLLM:
    """First call asks for create_ticket, second call gives the final answer."""
    def __init__(self):
        self.calls = 0
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            call = NS(id="c1", function=NS(name="create_ticket",
                      arguments=json.dumps({"issue": "VPN not working", "priority": "high"})))
            msg = NS(content=None, tool_calls=[call])
        else:
            msg = NS(content="I created ticket INC0001 for you.", tool_calls=None)
        return NS(choices=[NS(message=msg)])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "client", FakeLLM())
    monkeypatch.setattr(main, "TICKET_FILE", tmp_path / "tickets.json")
    main.sessions.clear()
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_creates_ticket(client):
    r = client.post("/chat", json={"message": "My VPN is not working"})
    assert r.status_code == 200
    assert "INC0001" in r.json()["reply"]
    assert client.get("/tickets").json()["INC0001"]["priority"] == "high"


def test_memory_is_kept(client):
    r = client.post("/chat", json={"message": "hi"})
    sid = r.json()["session_id"]
    assert len(main.sessions[sid]) > 2  # system + user + tool steps + reply


def test_unknown_ticket():
    assert "error" in main.get_ticket("INC9999")

def test_ticket_not_found(client):
    assert client.get("/tickets/INC9999").status_code == 404
