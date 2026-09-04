"""Tests for the FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_metrics_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data


@pytest.mark.asyncio
async def test_query_without_index(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"question": "What is this?"},
        )
    assert response.status_code == 200
    data = response.json()
    # Should return a message about no documents being ingested
    assert "answer" in data


@pytest.mark.asyncio
async def test_query_validation(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"question": ""},
        )
    # Pydantic validation should reject empty question
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_orchestrate_query_without_index(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/orchestrate/query",
            json={"question": "What is AdamW?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "No documents have been ingested" in data["answer"]


@pytest.mark.asyncio
async def test_orchestrate_validate_without_index(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/orchestrate/validate")
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data
    assert data["overall_status"] == "UNHEALTHY"


@pytest.mark.asyncio
async def test_orchestrate_ingest_no_files(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/orchestrate/ingest")
    # Missing required form-data files should return 422
    assert response.status_code == 422
