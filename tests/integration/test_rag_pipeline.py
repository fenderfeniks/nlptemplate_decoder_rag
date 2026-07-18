import json
import os

import pytest


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PAYLOADS_PATH = os.path.join(BASE_DIR, "data", "test_payloads.json")

with open(PAYLOADS_PATH, encoding="utf-8") as f:
    PAYLOADS = json.load(f)


@pytest.mark.parametrize("test_case", PAYLOADS["valid_requests"])
async def test_valid_requests(async_client, override_ml_deps, test_case):
    response = await async_client.post("/chat/generate", json=test_case["payload"])

    assert response.status_code == 200
    assert "answer" in response.json()


@pytest.mark.parametrize("test_case", PAYLOADS["edge_cases"])
async def test_edge_cases(async_client, override_ml_deps, test_case):
    response = await async_client.post("/chat/generate", json=test_case["payload"])

    assert response.status_code == 200


@pytest.mark.parametrize("test_case", PAYLOADS["ml_specific_risks"])
async def test_security_and_limits(async_client, override_ml_deps, test_case):
    response = await async_client.post("/chat/generate", json=test_case["payload"])

    assert response.status_code == 200
