"""
Persistent run history logger.
Appends one JSON line per run to data/results/run_history.jsonl.
Each entry survives model changes, re-runs, and fine-tuning experiments.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "data" / "results" / "run_history.jsonl"


def log_run(module: str, metrics: dict, params: dict = None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id":    uuid.uuid4().hex[:8],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module":    module,
        "params":    params or {},
        "metrics":   metrics,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  [log] run saved → {LOG_PATH.name}  (run_id={entry['run_id']})")


def print_history(module: str = None, last_n: int = 20) -> None:
    if not LOG_PATH.exists():
        print("No run history yet.")
        return
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(l) for l in lines if l.strip()]
    if module:
        entries = [e for e in entries if e["module"] == module]
    for e in entries[-last_n:]:
        metrics_str = "  ".join(f"{k}={v}" for k, v in e["metrics"].items())
        print(f"[{e['timestamp']}] {e['run_id']}  {e['module']:20s}  {metrics_str}")
