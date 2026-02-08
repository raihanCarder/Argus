from fastapi.testclient import TestClient
import pytest

import main


KEYWORDS = [
    "soc",
    "siem",
    "cyber",
    "security",
    "endpoint",
    "cloud",
    "data",
    "analytics",
    "health",
    "transit",
    "flood",
    "portal",
]


def _stub_embedding(text: str) -> list[float]:
    lowered = text.lower()
    return [1.0 if keyword in lowered else 0.0 for keyword in KEYWORDS]


@pytest.fixture(scope="module")
def client():
    main.get_embedding = _stub_embedding
    main.generate_match_reasoning = lambda *_args, **_kwargs: ""
    return TestClient(main.app)


def test_match_returns_results(client):
    r = client.post("/match", json={"startup_description": "AI SOC platform for government cybersecurity"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)

def test_cyber_startup_ranks_cyber_signal_high(client):
    r = client.post("/match", json={"startup_description": "SIEM SOC automation threat detection for city government"})
    assert r.status_code == 200
    data = r.json()

    # You need to adjust these keys once you know exact response format:
    matches = data.get("matches", [])
    assert isinstance(matches, list)
    assert len(matches) > 0

    top = matches[0]
    signal = top.get("signal", {})
    text = (signal.get("title", "") + " " + signal.get("description", "")).lower()
    assert ("soc" in text) or ("siem" in text) or ("cyber" in text)
