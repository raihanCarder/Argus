import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SIGNALS_PATH = DATA_DIR / "signals.json"
STARTUPS_PATH = DATA_DIR / "startups.json"


def _load_json(path: Path):
    assert path.exists(), f"Missing data file: {path}"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_signals_file_structure():
    signals = _load_json(SIGNALS_PATH)
    assert isinstance(signals, list)
    assert len(signals) > 0

    ids = set()
    for idx, signal in enumerate(signals):
        assert isinstance(signal, dict), f"Signal {idx} is not an object"

        signal_id = signal.get("id")
        assert isinstance(signal_id, int)
        assert signal_id not in ids
        ids.add(signal_id)

        for key in ["city", "state", "category", "title", "description", "timeline", "source"]:
            assert isinstance(signal.get(key), str)
            assert signal.get(key).strip() != ""

        budget = signal.get("budget")
        assert isinstance(budget, (int, float))
        assert budget >= 0

        lat = signal.get("lat")
        lng = signal.get("lng")
        assert isinstance(lat, (int, float))
        assert isinstance(lng, (int, float))
        assert -90 <= lat <= 90
        assert -180 <= lng <= 180

        keywords = signal.get("keywords")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(k, str) and k.strip() for k in keywords)

        stakeholders = signal.get("stakeholders")
        assert isinstance(stakeholders, list)
        assert len(stakeholders) > 0
        assert all(isinstance(s, str) and s.strip() for s in stakeholders)


def test_startups_file_structure():
    startups = _load_json(STARTUPS_PATH)
    assert isinstance(startups, list)
    assert len(startups) > 0

    ids = set()
    for idx, startup in enumerate(startups):
        assert isinstance(startup, dict), f"Startup {idx} is not an object"

        startup_id = startup.get("id")
        assert isinstance(startup_id, (int, str))
        assert startup_id not in ids
        ids.add(startup_id)

        name = startup.get("name")
        desc = startup.get("description")
        assert isinstance(name, str) and name.strip()
        assert isinstance(desc, str) and desc.strip()
