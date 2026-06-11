"""
Module 4: Classifier Training
TF-IDF + Logistic Regression + XGBoost (attack family classification)
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from db import connect

ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"


# ── Data loading (canonical — module5 and module4_v2 import from here) ───────

def expand_held_out(held_out: list, all_labels: set) -> list:
    """Include encoding_* sub-variants when 'encoding' is held out, etc."""
    expanded = set(held_out)
    for h in held_out:
        for label in all_labels:
            if label == h or label.startswith(h + "_"):
                expanded.add(label)
    return list(expanded)


def load_texts_labels() -> tuple[list, list]:
    """
    All labeled prompts as (texts, labels).
    Label priority: manual cluster family_label > mutation_op > attack_type.
    Note: labels are attack *techniques* (largely the mutation op that produced
    a prompt), not harm domains — see README "Limitations".
    """
    with connect() as con:
        rows = con.execute("""
            SELECT p.text,
                COALESCE(
                    CASE WHEN c.family_label IS NOT NULL
                              AND c.family_label != 'noise'
                              AND c.family_label NOT LIKE 'cluster_%'
                         THEN c.family_label ELSE NULL END,
                    p.mutation_op,
                    p.attack_type
                ) AS label
            FROM prompts p
            LEFT JOIN clusters c ON c.prompt_id = p.id
            WHERE p.attack_type IS NOT NULL OR p.mutation_op IS NOT NULL
        """).fetchall()

    texts  = [r["text"]  for r in rows if r["label"]]
    labels = [r["label"] for r in rows if r["label"]]
    return texts, labels


def load_labeled_data(held_out_families: list[str] = None) -> tuple[list, list, list, list]:
    """
    Returns (X_train, y_train, X_test, y_test).
    If held_out_families given, those families (sub-variants included) go to
    test only (open-set eval).
    """
    texts, labels = load_texts_labels()

    if held_out_families:
        held = set(expand_held_out(held_out_families, set(labels)))
        print(f"  Held-out expanded to: {sorted(held)}")
        train_x, train_y, test_x, test_y = [], [], [], []
        for t, l in zip(texts, labels):
            if l in held:
                test_x.append(t); test_y.append(l)
            else:
                train_x.append(t); train_y.append(l)
        return train_x, train_y, test_x, test_y

    from sklearn.model_selection import train_test_split
    from collections import Counter
    counts = Counter(labels)
    can_stratify = all(v >= 2 for v in counts.values())
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=42,
        stratify=labels if can_stratify else None,
    )
    return X_tr, y_tr, X_te, y_te


# ── Classifiers ───────────────────────────────────────────────────────────────

class _XGBPipeline:
    """Thin sklearn-compatible wrapper so dashboard can call .predict() / .predict_proba()."""
    def __init__(self, tfidf, xgb, le):
        self.tfidf = tfidf
        self.xgb   = xgb
        self.le    = le

    def predict(self, X):
        return self.le.inverse_transform(self.xgb.predict(self.tfidf.transform(X)))

    def predict_proba(self, X):
        return self.xgb.predict_proba(self.tfidf.transform(X))

    @property
    def classes_(self):
        return self.le.classes_


def train(X_train, y_train, X_test, y_test) -> dict:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score

    results = {}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Logistic Regression
    print("\n--- TF-IDF + Logistic Regression ---")
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    f1_lr = f1_score(y_test, y_pred_lr, average="macro", zero_division=0)
    print(classification_report(y_test, y_pred_lr, zero_division=0))
    print(f"Macro F1: {f1_lr:.4f}")
    results["logreg_f1"] = f1_lr
    with open(ARTIFACTS_DIR / "baseline_logreg.pkl", "wb") as f:
        pickle.dump(lr_pipe, f)

    # XGBoost
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder

        print("\n--- TF-IDF + XGBoost ---")
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        y_te_enc = le.transform(y_test)

        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
        X_tr_t = tfidf.fit_transform(X_train)
        X_te_t = tfidf.transform(X_test)

        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr_enc)

        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="mlogloss",
                             verbosity=0, tree_method="hist")
        xgb.fit(X_tr_t, y_tr_enc, sample_weight=sample_weights)
        y_pred_xgb = le.inverse_transform(xgb.predict(X_te_t))
        f1_xgb = f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)
        print(classification_report(y_test, y_pred_xgb, zero_division=0))
        print(f"Macro F1: {f1_xgb:.4f}")
        results["xgboost_f1"] = f1_xgb

        with open(ARTIFACTS_DIR / "baseline_xgb.pkl", "wb") as f:
            pickle.dump(_XGBPipeline(tfidf, xgb, le), f)

    except ImportError:
        print("  XGBoost not installed — skipping. pip install xgboost")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run(held_out: list[str] = None, **_) -> dict:
    print("=== Module 4: Classifier Training ===\n")

    X_train, y_train, X_test, y_test = load_labeled_data(held_out_families=held_out)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    if not X_train:
        print("No labeled data. Run Module 3 first.")
        return {}

    results = train(X_train, y_train, X_test, y_test)
    from run_logger import log_run
    log_run("module4_classify",
            metrics=results,
            params={"train_size": len(X_train), "test_size": len(X_test), "held_out": held_out})
    print("\nModule 4: Done.")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--held-out", nargs="+", default=None)
    args = p.parse_args()
    run(held_out=args.held_out)
