"""
Module 4 v2: Embedding-Based Classifier (parallel to TF-IDF baseline)

Uses intfloat/multilingual-e5-base instead of TF-IDF features.
Benefits:
  - Handles non-English prompts properly (no longer trivially anomalous)
  - Captures semantic similarity TF-IDF misses (paraphrase, token_smuggling)
  - k-NN distance gives a better novelty signal than max-confidence

Run alongside module4_classify.py to compare:
  python src/module4_classify_v2.py
  python src/module4_classify.py
Then compare logreg_f1 vs emb_logreg_f1 and AUROC results.
"""
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from db import connect
from run_logger import log_run

ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"
RESULTS_DIR   = Path(__file__).parent.parent / "data" / "results"
EMB_MODEL     = "intfloat/multilingual-e5-base"


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)


def embed(texts: list, embedder=None, batch_size: int = 128) -> np.ndarray:
    if embedder is None:
        embedder = get_embedder()
    # multilingual-e5 needs "query: " prefix for retrieval/classification
    prefixed = [f"query: {t}" for t in texts]
    return embedder.encode(prefixed, batch_size=batch_size,
                           normalize_embeddings=True,
                           show_progress_bar=True).astype(np.float32)


# ── Data loading (reuse module4 logic) ───────────────────────────────────────

def load_labeled_data(held_out_families: list = None):
    from module4_classify import load_labeled_data as _load
    return _load(held_out_families=held_out_families)


# ── k-NN novelty detector ─────────────────────────────────────────────────────

class KNNNoveltyDetector:
    """
    Novelty score = mean distance to k nearest training neighbors.
    High distance = likely novel/unseen family.
    """
    def __init__(self, k: int = 5):
        self.k = k
        self.train_embs = None

    def fit(self, train_embs: np.ndarray):
        self.train_embs = train_embs
        return self

    def novelty_scores(self, test_embs: np.ndarray) -> np.ndarray:
        # Cosine similarity → distance = 1 - sim (embeddings are normalized)
        sims = test_embs @ self.train_embs.T          # (n_test, n_train)
        top_k_sims = np.partition(sims, -self.k, axis=1)[:, -self.k:]
        mean_sim = top_k_sims.mean(axis=1)
        return 1.0 - mean_sim                         # higher = more novel


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_train: list, y_train: list, X_test: list, y_test: list) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, classification_report
    from sklearn.model_selection import train_test_split
    from collections import Counter

    results = {}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nEmbedding model: {EMB_MODEL}")
    embedder = get_embedder()

    print("Embedding training set...")
    X_tr_emb = embed(X_train, embedder)
    print("Embedding test set...")
    X_te_emb = embed(X_test, embedder)

    # LogReg on embeddings
    print("\n--- Multilingual-E5 + Logistic Regression ---")
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_tr_emb, y_train)
    preds = clf.predict(X_te_emb)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(classification_report(y_test, preds, zero_division=0))
    print(f"Macro F1: {f1:.4f}")
    results["emb_logreg_f1"] = round(f1, 4)

    # Save classifier + embedder name
    with open(ARTIFACTS_DIR / "emb_logreg.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(ARTIFACTS_DIR / "emb_model_name.txt", "w") as f:
        f.write(EMB_MODEL)

    # k-NN novelty detector
    print("\n--- k-NN Novelty Detector ---")
    knn = KNNNoveltyDetector(k=5)
    knn.fit(X_tr_emb)
    with open(ARTIFACTS_DIR / "knn_novelty.pkl", "wb") as f:
        pickle.dump(knn, f)

    # Save train embeddings for AUROC comparison in module5
    np.save(ARTIFACTS_DIR / "train_embs.npy", X_tr_emb)
    np.save(ARTIFACTS_DIR / "test_embs.npy", X_te_emb)

    print("Artifacts saved.")
    return results


# ── AUROC comparison ──────────────────────────────────────────────────────────

def compare_auroc(holdout_sets: list = None) -> dict:
    """
    Compare TF-IDF max-confidence vs k-NN distance as novelty signals,
    across every holdout set (not just the most favorable one).

    Fairness: BOTH methods are trained on the same holdout-excluded 80%
    split. The previous version reused baseline_logreg.pkl — trained on a
    random split that INCLUDED the held-out families — so its confident
    (correct) predictions on "unseen" prompts pushed TF-IDF AUROC below
    chance and inflated the gap.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from collections import Counter

    from module4_classify import load_texts_labels, expand_held_out
    if holdout_sets is None:
        from module5_evaluate import MULTI_HOLDOUT_SETS
        holdout_sets = MULTI_HOLDOUT_SETS

    texts, labels = load_texts_labels()
    all_label_set = set(labels)

    embedder = get_embedder()
    print(f"Embedding full corpus once ({len(texts)} prompts)...")
    all_embs = embed(texts, embedder)

    table = {}
    for hset in holdout_sets:
        key = "+".join(hset)
        held = set(expand_held_out(hset, all_label_set))
        seen_idx   = [i for i, l in enumerate(labels) if l not in held]
        unseen_idx = [i for i, l in enumerate(labels) if l in held]
        if not seen_idx or not unseen_idx:
            print(f"  {key}: skipped (degenerate split)")
            continue

        y_seen = [labels[i] for i in seen_idx]
        counts = Counter(y_seen)
        can_strat = all(v >= 2 for v in counts.values())
        tr_idx, val_idx = train_test_split(
            seen_idx, test_size=0.2, random_state=42,
            stratify=y_seen if can_strat else None,
        )

        y_bin = [1] * len(unseen_idx) + [0] * len(val_idx)

        # TF-IDF max-confidence — retrained without the held-out families
        tfidf_clf = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000,
                                      sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0,
                                         class_weight="balanced")),
        ])
        tfidf_clf.fit([texts[i] for i in tr_idx], [labels[i] for i in tr_idx])
        max_conf = tfidf_clf.predict_proba(
            [texts[i] for i in unseen_idx + val_idx]).max(axis=1)
        auroc_tfidf = float(roc_auc_score(y_bin, 1.0 - max_conf))

        # Embedding k-NN distance — same train split
        knn = KNNNoveltyDetector(k=5).fit(all_embs[tr_idx])
        scores = np.concatenate([knn.novelty_scores(all_embs[unseen_idx]),
                                 knn.novelty_scores(all_embs[val_idx])])
        auroc_knn = float(roc_auc_score(np.array(y_bin), scores))

        winner = "knn" if auroc_knn > auroc_tfidf else "tfidf"
        table[key] = {
            "held_out": hset,
            "n_train": len(tr_idx),
            "n_seen_val": len(val_idx),
            "n_unseen": len(unseen_idx),
            "tfidf_auroc": round(auroc_tfidf, 4),
            "knn_auroc": round(auroc_knn, 4),
            "better_method": winner,
        }
        print(f"  {key:35s} tfidf={auroc_tfidf:.4f}  knn={auroc_knn:.4f}  → {winner}")

    return {"holdout_table": table}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(held_out: list = None, compare: bool = True) -> dict:
    print("=== Module 4 v2: Embedding-Based Classifier ===\n")

    X_train, y_train, X_test, y_test = load_labeled_data()
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    if not X_train:
        print("No labeled data. Run Module 3 first.")
        return {}

    results = train(X_train, y_train, X_test, y_test)

    if compare:
        print("\n=== AUROC Comparison: TF-IDF vs Embedding k-NN (all holdouts) ===")
        auroc_results = compare_auroc(holdout_sets=[held_out] if held_out else None)
        results.update(auroc_results)

        out = RESULTS_DIR / "auroc_comparison.json"
        out.write_text(__import__("json").dumps(auroc_results, indent=2), encoding="utf-8")
        print(f"\nComparison saved to {out}")

    log_run("module4_classify_v2",
            metrics=results,
            params={"emb_model": EMB_MODEL, "held_out": held_out,
                    "train_size": len(X_train), "test_size": len(X_test)})
    print("\nModule 4 v2: Done.")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Embedding-based classifier (parallel to TF-IDF baseline)")
    p.add_argument("--held-out", nargs="+", default=None)
    p.add_argument("--no-compare", action="store_true", help="Skip AUROC comparison")
    args = p.parse_args()
    run(held_out=args.held_out, compare=not args.no_compare)
