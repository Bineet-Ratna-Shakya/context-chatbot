from __future__ import annotations

import app.main as main_module


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_message(client):
    response = client.post("/chat", json={"message": "", "thread_id": "t"})
    assert response.status_code == 422


def test_chat_returns_answer_and_sources(client):
    response = client.post("/chat", json={"message": "What is LangGraph?", "thread_id": "t"})
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "hello"
    assert body["sources"][0]["title"] == "LangGraph"
    assert body["sources"][0]["page"] is None


def test_chat_maps_graph_error_to_500(failing_client):
    response = failing_client.post("/chat", json={"message": "hi", "thread_id": "t"})
    assert response.status_code == 500
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_rate_limit_returns_429(monkeypatch, client):
    monkeypatch.setattr(main_module, "RATE_LIMIT_PER_MINUTE", 2)
    payload = {"message": "hi", "thread_id": "t"}
    assert client.post("/chat", json=payload).status_code == 200
    assert client.post("/chat", json=payload).status_code == 200
    assert client.post("/chat", json=payload).status_code == 429
