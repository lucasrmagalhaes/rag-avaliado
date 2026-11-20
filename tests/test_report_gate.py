import json
from pathlib import Path

from app.metrics import quality_failures

ROOT = Path(__file__).resolve().parents[1]


def test_measured_report_meets_threshold():
    report = json.loads((ROOT / "reports" / "eval_latest.json").read_text(encoding="utf-8"))
    thresholds = json.loads((ROOT / "data" / "eval" / "thresholds.json").read_text(encoding="utf-8"))
    assert report["n_questions"] >= 50
    assert report["n_chunks"] > 200
    assert report["llm_effective"] == "mock"
    assert report["embedding_model"]
    assert report["cost"]["billed"] is None
    assert set(report["modes"]) == {"vector", "hybrid", "hybrid_rerank"}
    assert quality_failures(report, thresholds) == []
    hybrid = report["modes"]["hybrid"]
    vector = report["modes"]["vector"]
    assert hybrid["hit_rate_at_5"] > vector["hit_rate_at_5"]
