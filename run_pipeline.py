"""
Master pipeline runner. Executes all 6 modules in order.
Run individual modules by passing --module N, or run all with no args.

Usage:
  python run_pipeline.py                     # full pipeline
  python run_pipeline.py --module 1          # ingest only
  python run_pipeline.py --module 2          # mutate only
  python run_pipeline.py --module 2 --no-llm # fast mutations only
  python run_pipeline.py --module 3          # cluster
  python run_pipeline.py --module 4 --stage baseline
  python run_pipeline.py --module 5
  streamlit run src/module6_dashboard.py     # dashboard (separate)
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser(description="Jailbreak Genome Lab — Pipeline Runner")
    p.add_argument("--module", type=int, default=None, choices=[1,2,3,4,5],
                   help="Run a single module (1-5). Omit for full pipeline.")

    # Module 1 args
    p.add_argument("--reset",     action="store_true", help="[M1] Clear DB before ingesting")
    p.add_argument("--no-dedup",  action="store_true", help="[M1] Skip deduplication")

    # Module 2 args
    p.add_argument("--mutations",  type=int,   default=4,    help="[M2] Mutations per seed")
    p.add_argument("--no-llm",     action="store_true",       help="[M2] Skip LLM mutations (faster)")
    p.add_argument("--limit",      type=int,   default=None,  help="[M2] Limit seeds for testing")

    # Module 3 args
    p.add_argument("--min-cluster-size", type=int, default=15, help="[M3] HDBSCAN min cluster size")

    # Module 4 args
    p.add_argument("--stage",     choices=["baseline","qlora","both"], default="both", help="[M4]")
    p.add_argument("--held-out",  nargs="+", default=None,  help="[M4/M5] Families to hold out")
    p.add_argument("--epochs",    type=int,  default=3,      help="[M4] QLoRA epochs")

    # Module 5 args
    p.add_argument("--judge-sample", type=int, default=50, help="[M5] Prompts to judge with 8B")

    args = p.parse_args()

    run_all = args.module is None

    if run_all or args.module == 1:
        from module1_ingest import run as ingest
        from db import DB_PATH
        if args.reset:
            DB_PATH.unlink(missing_ok=True)
            print("DB cleared.\n")
        ingest(dedup=not args.no_dedup)

    if run_all or args.module == 2:
        from module2_mutate import run as mutate
        mutate(mutations_per_seed=args.mutations, use_llm=not args.no_llm, limit=args.limit)

    if run_all or args.module == 3:
        from module3_cluster import run as cluster
        cluster(min_cluster_size=args.min_cluster_size)

    if run_all or args.module == 4:
        from module4_classify import run as classify
        classify(stage=args.stage, held_out=args.held_out, epochs=args.epochs)

    if run_all or args.module == 5:
        from module5_evaluate import run as evaluate
        evaluate(held_out=args.held_out, judge_sample=args.judge_sample)

    if run_all:
        print("\n=== Full pipeline complete ===")
        print("Start the dashboard with:")
        print("  streamlit run src/module6_dashboard.py")


if __name__ == "__main__":
    main()
