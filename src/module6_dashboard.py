"""
Module 6: Drift & Robustness Dashboard (Streamlit)
Run: streamlit run src/module6_dashboard.py

Features:
- Upload prompt → predicted attack family + confidence
- Nearest seed in embedding space
- Mutation distance from known attacks
- Live UMAP plot
- Novel attack flagging
"""
import sys
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

ARTIFACTS_DIR  = Path(__file__).parent.parent / "data" / "artifacts"
CLASSIFIER_DIR = Path(__file__).parent.parent / "data" / "classifier"


# Cached loaders 

@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_classifier():
    # XGBoost outperforms LogReg on encoding subtypes — prefer it
    for name in ("baseline_xgb.pkl", "baseline_logreg.pkl"):
        path = ARTIFACTS_DIR / name
        if path.exists():
            return pickle.loads(path.read_bytes())
    return None


@st.cache_resource
def load_hdbscan():
    path = ARTIFACTS_DIR / "hdbscan_model.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    return None


@st.cache_resource
def load_umap_reducer():
    path = ARTIFACTS_DIR / "umap_reducer.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    return None


@st.cache_data
def load_umap_points() -> list[dict]:
    path = ARTIFACTS_DIR / "umap_points.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


@st.cache_data
def load_corpus_embeddings():
    """Load all stored embeddings for nearest-neighbor search."""
    from db import connect
    with connect() as con:
        rows = con.execute(
            "SELECT id, text, embedding FROM prompts WHERE embedding IS NOT NULL LIMIT 5000"
        ).fetchall()
    if not rows:
        return [], [], np.array([])
    ids   = [r["id"]   for r in rows]
    texts = [r["text"] for r in rows]
    embs  = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return ids, texts, embs


# Analysis helpers

def analyze_prompt(text: str) -> dict:
    result = {}

    embedder = load_embedder()
    emb = embedder.encode([text], normalize_embeddings=True)[0].astype(np.float32)
    result["embedding"] = emb

    # Classifier prediction
    clf = load_classifier()
    if clf is not None:
        pred = clf.predict([text])[0]
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba([text])[0]
            conf = float(proba.max())
        else:
            conf = 0.0
        result["family"] = pred
        result["confidence"] = conf
        result["is_novel"] = conf < 0.4  # low confidence → flag as novel
    else:
        result["family"] = "classifier not trained"
        result["confidence"] = 0.0
        result["is_novel"] = True

    # Nearest neighbor
    ids, texts, corpus_embs = load_corpus_embeddings()
    if len(corpus_embs) > 0:
        sims = corpus_embs @ emb
        top_idx = int(np.argmax(sims))
        result["nearest_text"] = texts[top_idx]
        result["nearest_sim"]  = float(sims[top_idx])

        # Mutation distance: hop count from seed
        from db import connect
        with connect() as con:
            nearest_id = ids[top_idx]
            hops = 0
            cur_id = nearest_id
            for _ in range(10):
                row = con.execute("SELECT parent_id FROM prompts WHERE id = ?", (cur_id,)).fetchone()
                if row is None or row["parent_id"] is None:
                    break
                cur_id = row["parent_id"]
                hops += 1
        result["mutation_distance"] = hops

    # UMAP coordinates for new point
    reducer = load_umap_reducer()
    if reducer is not None:
        coords = reducer.transform([emb])
        result["umap_x"] = float(coords[0, 0])
        result["umap_y"] = float(coords[0, 1])

    return result


# Results tab

def _render_results_tab():
    from visualize import (plot_auroc_history, plot_f1_history,
                           plot_corpus_growth, plot_mutation_distance_curve,
                           plot_fitness_evolution, PLOTS_DIR)

    st.subheader("Run History")

    # Raw log table
    log_path = PLOTS_DIR.parent / "run_history.jsonl"
    if log_path.exists():
        import json
        entries = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if entries:
            import pandas as pd
            rows = []
            for e in entries:
                row = {"run_id": e["run_id"], "timestamp": e["timestamp"], "module": e["module"]}
                row.update({f"metric:{k}": v for k, v in e.get("metrics", {}).items()
                            if isinstance(v, (int, float))})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    st.subheader("Charts")

    if st.button("Regenerate all charts"):
        plot_auroc_history()
        plot_f1_history()
        plot_corpus_growth()
        plot_mutation_distance_curve()
        plot_fitness_evolution()
        st.success("Charts regenerated.")

    chart_defs = [
        ("latest_auroc_history.png",          "AUROC Across Runs"),
        ("latest_f1_history.png",             "Classifier F1 Across Runs"),
        ("latest_corpus_growth.png",          "Corpus Growth"),
        ("latest_mutation_distance_curve.png","Detection Rate vs Mutation Distance"),
        ("latest_fitness_evolution.png",      "Fitness Evolution (Module 7)"),
    ]

    for filename, title in chart_defs:
        path = PLOTS_DIR / filename
        if path.exists():
            st.caption(title)
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"{title} — not generated yet. Click 'Regenerate all charts'.")


# Dashboard UI

def main():
    st.set_page_config(page_title="Jailbreak Genome Lab", layout="wide")
    st.title("Jailbreak Genome Lab — Drift & Robustness Dashboard")

    tab1, tab2 = st.tabs(["Prompt Analysis", "Results & History"])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Analyze a Prompt")
            prompt_input = st.text_area("Paste a prompt to analyze:", height=150,
                                        placeholder="Enter any prompt here...")
            analyze_btn = st.button("Analyze", type="primary")

            if analyze_btn and prompt_input.strip():
                with st.spinner("Analyzing..."):
                    result = analyze_prompt(prompt_input.strip())

                family   = result.get("family", "unknown")
                conf     = result.get("confidence", 0.0)
                is_novel = result.get("is_novel", False)

                if is_novel:
                    st.error("NOVEL ATTACK DETECTED — Human review needed")
                else:
                    st.success(f"Known family: **{family}**")

                m1, m2, m3 = st.columns(3)
                m1.metric("Attack Family", family)
                m2.metric("Confidence", f"{conf:.0%}")
                m3.metric("Mutation Distance", result.get("mutation_distance", "N/A"))

                if "nearest_text" in result:
                    st.subheader("Nearest Known Prompt")
                    st.write(f"Similarity: `{result['nearest_sim']:.3f}`")
                    st.info(result["nearest_text"])

        with col2:
            st.subheader("Attack Family Landscape (UMAP)")
            points = load_umap_points()

            if not points:
                st.warning("No UMAP data yet. Run Module 3 first.")
            else:
                try:
                    import plotly.express as px
                    import pandas as pd

                    df = pd.DataFrame(points)
                    if analyze_btn and prompt_input.strip() and "umap_x" in result:
                        new_row = pd.DataFrame([{
                            "x": result["umap_x"], "y": result["umap_y"],
                            "family": f"NEW: {result.get('family','?')}",
                            "text_preview": prompt_input[:80],
                            "source": "input",
                        }])
                        df = pd.concat([df, new_row], ignore_index=True)

                    fig = px.scatter(
                        df, x="x", y="y", color="family",
                        hover_data=["text_preview", "source"],
                        title="Prompt Embedding Space",
                        width=650, height=550,
                        opacity=0.6,
                    )
                    fig.update_traces(marker_size=4)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.warning("pip install plotly pandas for interactive chart.")
                    st.write(f"{len(points)} points in corpus.")

    with tab2:
        _render_results_tab()

    # Corpus stats sidebar
    st.sidebar.header("Corpus Stats")
    try:
        from db import count_prompts
        st.sidebar.metric("Total Prompts", count_prompts())
    except Exception:
        pass

    points = load_umap_points()
    if points:
        from collections import Counter
        fam_counts = Counter(p["family"] for p in points)
        st.sidebar.subheader("Family Distribution")
        for fam, cnt in fam_counts.most_common():
            st.sidebar.write(f"`{fam}`: {cnt}")

    try:
        from db import connect
        with connect() as con:
            n_seeds   = con.execute("SELECT COUNT(*) FROM prompts WHERE parent_id IS NULL").fetchone()[0]
            n_mutated = con.execute("SELECT COUNT(*) FROM prompts WHERE parent_id IS NOT NULL").fetchone()[0]
        st.sidebar.metric("Seeds", n_seeds)
        st.sidebar.metric("Mutations", n_mutated)
    except Exception:
        pass


if __name__ == "__main__":
    main()
