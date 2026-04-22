"""API-level tests using httpx ASGITransport (no network, no server)."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from llm_gateway.config.settings import get_settings
from llm_gateway.main import create_app


@pytest.fixture
def app():
    get_settings.cache_clear()
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "echo" in body["providers"]


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    r = await client.get("/metrics")
    assert r.status_code == 200
    assert "llmgw_requests_total" in r.text


@pytest.mark.asyncio
async def test_complete_basic(client):
    r = await client.post("/v1/complete", json={"prompt": "Hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["completion"].startswith("[echo/")
    assert body["metadata"]["cache_status"] == "miss"
    assert body["metadata"]["provider"] == "echo"


@pytest.mark.asyncio
async def test_complete_empty_prompt_rejected(client):
    r = await client.post("/v1/complete", json={"prompt": ""})
    assert r.status_code == 400
    body = r.json()
    assert body["detail"]["error"] == "empty_prompt"


@pytest.mark.asyncio
async def test_bypass_optimization(client):
    long_prompt = "Could you please kindly summarize this document for me."
    r = await client.post(
        "/v1/complete",
        # Bypass BOTH coaching and optimization so the sent prompt is preserved verbatim
        json={
            "prompt": long_prompt,
            "bypass_optimization": True,
            "bypass_coaching": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"]["optimized_prompt"] == long_prompt
    assert body["metadata"]["optimization"]["applied"] is False


@pytest.mark.asyncio
async def test_oversize_with_rejection(monkeypatch, client):
    monkeypatch.setenv("LLMGW_MAX_PROMPT_TOKENS", "5")
    monkeypatch.setenv("LLMGW_REJECT_OVERSIZE", "true")
    # Need a fresh app so new settings take effect
    get_settings.cache_clear()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.post(
            "/v1/complete",
            json={"prompt": "this prompt is longer than five tokens for sure"},
        )
    assert r.status_code == 413
    assert r.json()["detail"]["error"] == "oversize_prompt"


@pytest.mark.asyncio
async def test_stats_endpoint(client):
    await client.post("/v1/complete", json={"prompt": "hello"})
    r = await client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["totals"]["requests"] >= 1
    assert "cache" in body and "latency" in body
    assert isinstance(body["recent"], list)
    assert len(body["recent"]) >= 1


@pytest.mark.asyncio
async def test_dashboard_served(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "LLM" in r.text and "Gateway" in r.text
