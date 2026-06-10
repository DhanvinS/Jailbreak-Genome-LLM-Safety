"""
Corpus cleaning utility.
Run once after ingestion/mutation to:
  1. Delete refusal strings that got written as prompts (LLM refused a mutation)
  2. Populate the language column using langdetect
Run: python src/clean_corpus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from db import connect

# Strings that indicate the LLM refused rather than mutated
REFUSAL_PATTERNS = [
    "i can't", "i cannot", "i'm unable", "i'm not able",
    "as an ai", "as a language model", "i apologize",
    "i'm sorry, but", "i must decline", "that's not something i",
    "i won't", "i will not provide",
]


def delete_refusals(dry_run: bool = False) -> int:
    with connect() as con:
        rows = con.execute("SELECT id, text FROM prompts WHERE source = 'synthetic'").fetchall()

    to_delete = []
    for row in rows:
        text_lower = row["text"].lower().strip()
        if any(text_lower.startswith(p) or p in text_lower[:80] for p in REFUSAL_PATTERNS):
            to_delete.append(row["id"])

    print(f"Found {len(to_delete)} refusal strings in synthetic prompts.")
    if to_delete:
        print("  Examples:")
        with connect() as con:
            for pid in to_delete[:3]:
                t = con.execute("SELECT text FROM prompts WHERE id=?", (pid,)).fetchone()
                print(f"    [{pid}] {t['text'][:80]}")

    if not dry_run and to_delete:
        with connect() as con:
            con.executemany("DELETE FROM prompts WHERE id = ?", [(i,) for i in to_delete])
        print(f"Deleted {len(to_delete)} refusal prompts.")
    elif dry_run:
        print("Dry run — nothing deleted.")

    return len(to_delete)


def populate_language(batch_size: int = 500) -> int:
    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        print("langdetect not installed. Run: pip install langdetect")
        return 0

    with connect() as con:
        rows = con.execute(
            "SELECT id, text FROM prompts WHERE language = 'en' OR language IS NULL"
        ).fetchall()

    updated = 0
    for i, row in enumerate(rows):
        try:
            lang = detect(row["text"])
        except Exception:
            lang = "en"

        if lang != "en":
            with connect() as con:
                con.execute("UPDATE prompts SET language = ? WHERE id = ?",
                            (lang, row["id"]))
            updated += 1

        if (i + 1) % batch_size == 0:
            print(f"  [{i+1}/{len(rows)}] language tags updated so far: {updated}")

    print(f"Language column updated for {updated} non-English prompts.")
    return updated


def run(dry_run: bool = False) -> None:
    print("=== Corpus Cleaning ===\n")

    print("Step 1: Remove LLM refusals from synthetic prompts")
    n_deleted = delete_refusals(dry_run=dry_run)

    print("\nStep 2: Populate language column")
    n_updated = populate_language()

    print(f"\nDone. Deleted {n_deleted} refusals, updated {n_updated} language tags.")

    from run_logger import log_run
    log_run("clean_corpus",
            metrics={"refusals_deleted": n_deleted, "language_tags_updated": n_updated},
            params={"dry_run": dry_run})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = p.parse_args()
    run(dry_run=args.dry_run)
