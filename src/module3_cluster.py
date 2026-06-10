"""
Module 3: Attack Family Clustering
- Embeds all prompts with MiniLM
- Clusters with HDBSCAN
- Reduces to 2D with UMAP for visualization
- Saves cluster assignments + UMAP coords back to SQLite
"""
import sys
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from db import connect, DB_PATH

ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"

# Cluster ID → human-readable family label (assigned after inspecting clusters)
# Override this map after running and reviewing the UMAP plot
CLUSTER_LABEL_MAP: dict[int, str] = {
    -1: "noise",
}

KNOWN_FAMILIES = [
    "roleplay", "encoding", "social_engineering", "persona_hijack",
    "token_smuggling", "indirect_injection", "multilingual", "paraphrase",
    "direct_harm", "obfuscation",
]


def embed_all(batch_size: int = 256) -> tuple[list[int], np.ndarray]:
    """Embed all prompts without embeddings. Returns (ids, embeddings)."""
    from sentence_transformers import SentenceTransformer

    with connect() as con:
        rows = con.execute("SELECT id, text FROM prompts WHERE embedding IS NULL").fetchall()

    if not rows:
        # All already embedded — load from DB
        with connect() as con:
            rows = con.execute("SELECT id, embedding FROM prompts WHERE embedding IS NOT NULL").fetchall()
        ids = [r["id"] for r in rows]
        embeddings = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return ids, embeddings

    print(f"Embedding {len(rows)} prompts with MiniLM...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [r["text"] for r in rows]
    ids   = [r["id"]   for r in rows]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    # Persist embeddings to DB
    with connect() as con:
        for row_id, emb in zip(ids, embeddings):
            con.execute("UPDATE prompts SET embedding = ? WHERE id = ?",
                        (emb.tobytes(), row_id))
    print("Embeddings saved to DB.")
    return ids, embeddings


def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 15) -> np.ndarray:
    try:
        import hdbscan
    except ImportError:
        raise ImportError("pip install hdbscan")

    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    print(f"Found {n_clusters} clusters, {noise_pct:.1f}% noise points.")

    # Save clusterer for later (dashboard inference)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "hdbscan_model.pkl", "wb") as f:
        pickle.dump(clusterer, f)

    return labels


def reduce_umap(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("pip install umap-learn")

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42, verbose=False)
    coords = reducer.fit_transform(embeddings).astype(np.float32)

    with open(ARTIFACTS_DIR / "umap_reducer.pkl", "wb") as f:
        pickle.dump(reducer, f)

    return coords


def infer_cluster_labels(ids: list[int], labels: np.ndarray) -> dict[int, str]:
    """
    Auto-assign family labels by taking the plurality attack_type within each cluster.
    Falls back to CLUSTER_LABEL_MAP for manual overrides, then generic names.
    """
    from collections import Counter
    with connect() as con:
        attack_map = {r["id"]: r["attack_type"] for r in
                      con.execute("SELECT id, attack_type FROM prompts").fetchall()}

    cluster_to_types: dict[int, list[str]] = {}
    for row_id, label in zip(ids, labels):
        cluster_id = int(label)
        atype = attack_map.get(row_id) or ""
        cluster_to_types.setdefault(cluster_id, []).append(atype)

    inferred: dict[int, str] = {-1: "noise"}
    for cluster_id, types in cluster_to_types.items():
        if cluster_id == -1:
            continue
        # Manual override takes precedence
        if cluster_id in CLUSTER_LABEL_MAP:
            inferred[cluster_id] = CLUSTER_LABEL_MAP[cluster_id]
        else:
            counts = Counter(t for t in types if t)
            inferred[cluster_id] = counts.most_common(1)[0][0] if counts else f"cluster_{cluster_id}"

    print(f"Cluster labels: {inferred}")
    return inferred


def save_cluster_assignments(ids: list[int], labels: np.ndarray, coords: np.ndarray) -> None:
    label_map = infer_cluster_labels(ids, labels)
    with connect() as con:
        for row_id, label, (x, y) in zip(ids, labels, coords):
            family = label_map.get(int(label), f"cluster_{label}")
            con.execute(
                "INSERT OR REPLACE INTO clusters (prompt_id, cluster_id, family_label, umap_x, umap_y) "
                "VALUES (?, ?, ?, ?, ?)",
                (row_id, int(label), family, float(x), float(y))
            )
    print(f"Saved {len(ids)} cluster assignments.")


def export_umap_json(ids: list[int], labels: np.ndarray, coords: np.ndarray,
                     label_map: dict[int, str] = None) -> Path:
    """Export UMAP data as JSON for the Streamlit dashboard."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if label_map is None:
        label_map = CLUSTER_LABEL_MAP

    with connect() as con:
        texts = {r["id"]: r["text"] for r in con.execute("SELECT id, text FROM prompts").fetchall()}
        sources = {r["id"]: r["source"] for r in con.execute("SELECT id, source FROM prompts").fetchall()}
        attack_types = {r["id"]: r["attack_type"] for r in con.execute("SELECT id, attack_type FROM prompts").fetchall()}

    points = []
    for row_id, label, (x, y) in zip(ids, labels, coords):
        points.append({
            "id": row_id,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "cluster": int(label),
            "family": label_map.get(int(label), f"cluster_{label}"),
            "source": sources.get(row_id, ""),
            "attack_type": attack_types.get(row_id, ""),
            "text_preview": texts.get(row_id, "")[:80],
        })

    out = ARTIFACTS_DIR / "umap_points.json"
    out.write_text(json.dumps(points, ensure_ascii=False), encoding="utf-8")
    print(f"UMAP JSON exported to {out}")
    return out


def plot_umap_static(ids: list[int], labels: np.ndarray, coords: np.ndarray,
                     label_map: dict[int, str] = None) -> None:
    """Generate a static PNG scatter plot."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not installed — skipping static plot.")
        return

    unique = sorted(set(labels))
    colors = cm.tab20(np.linspace(0, 1, max(len(unique), 1)))
    color_map = {lab: colors[i] for i, lab in enumerate(unique)}

    if label_map is None:
        label_map = CLUSTER_LABEL_MAP

    fig, ax = plt.subplots(figsize=(14, 10))
    for lab in unique:
        mask = labels == lab
        family = label_map.get(int(lab), f"cluster_{lab}")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[lab]], label=family, s=4, alpha=0.6)

    ax.set_title("Jailbreak Attack Families — UMAP Projection")
    ax.legend(markerscale=4, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()

    out = ARTIFACTS_DIR / "umap_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Static UMAP plot saved to {out}")


def run(min_cluster_size: int = 15) -> None:
    print("=== Module 3: Clustering ===\n")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ids, embeddings = embed_all()
    print(f"Working with {len(ids)} embeddings.")

    labels = cluster_hdbscan(embeddings, min_cluster_size=min_cluster_size)
    coords = reduce_umap(embeddings)

    label_map = infer_cluster_labels(ids, labels)
    save_cluster_assignments(ids, labels, coords)
    export_umap_json(ids, labels, coords, label_map=label_map)
    plot_umap_static(ids, labels, coords, label_map=label_map)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = round(sum(1 for l in labels if l == -1) / max(len(labels), 1) * 100, 1)
    from run_logger import log_run
    log_run("module3_cluster",
            metrics={"n_embeddings": len(ids), "n_clusters": n_clusters, "noise_pct": noise_pct},
            params={"min_cluster_size": min_cluster_size})
    print("\nModule 3: Done.")
    print("Next step: review umap_plot.png and update CLUSTER_LABEL_MAP in this file.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--min-cluster-size", type=int, default=15)
    args = p.parse_args()
    run(min_cluster_size=args.min_cluster_size)
