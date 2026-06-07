# Jailbreak Genome Lab

A local, zero-cost research system for studying LLM robustness by evolving and classifying jailbreak prompts. Built as a portfolio project targeting AI Safety / Integrity roles.

## What it does

1. **Ingests** real jailbreak datasets (JailbreakBench, LibrAI/do-not-answer, built-in seeds)
2. **Mutates** prompts using 5 strategies: encoding, roleplay, indirect injection, paraphrase, multilingual
3. **Clusters** the full corpus with HDBSCAN + UMAP to discover attack family structure
4. **Classifies** attack families using TF-IDF + Logistic Regression + XGBoost
5. **Evaluates** open-set recognition — can the classifier flag *unseen* attack families as anomalous?
6. **Visualizes** everything in a live Streamlit dashboard

Key result: **0.87 AUROC** on held-out attack families (encoding + multilingual) using a TF-IDF + XGBoost classifier.

## Architecture

```
data/
  corpus.db          ← SQLite: prompts, clusters, evals
  artifacts/
    hdbscan_model.pkl
    umap_reducer.pkl
    umap_points.json
    umap_plot.png
    baseline_logreg.pkl
    baseline_xgb.pkl
  results/
    eval_report.json

src/
  db.py              ← SQLite schema + helpers
  llm_runner.py      ← Ollama HTTP client (llama3.2:3b + llama3.1:8b)
  module1_ingest.py  ← Dataset download + dedup
  module2_mutate.py  ← 5 mutation strategies
  module3_cluster.py ← HDBSCAN + UMAP
  module4_classify.py← TF-IDF + LogReg + XGBoost
  module5_evaluate.py← Open-set AUROC + mutation distance curve
  module6_dashboard.py← Streamlit UI

run_pipeline.py      ← Master orchestrator
requirements.txt
```

## Setup

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Ollama (local LLM runner)

Only needed for Module 2 LLM mutations and Module 5 judge evaluation. Download from [ollama.com](https://ollama.com) and pull the two models:

```bash
ollama pull llama3.2:3b                      # mutation engine — fast, high volume
ollama pull llama3.1:8b-instruct-q4_K_M     # judge evaluator — quality
```

Start Ollama before running those modules:

```bash
ollama serve
```

## Running the pipeline

```bash
# Full pipeline (sequential)
python run_pipeline.py

# Or module by module:
python run_pipeline.py --module 1                   # ingest ~1k prompts
python run_pipeline.py --module 2 --no-llm          # fast mutations (no Ollama needed)
python run_pipeline.py --module 2                   # add LLM mutations (needs Ollama)
python run_pipeline.py --module 3                   # cluster + UMAP
python run_pipeline.py --module 4                   # train TF-IDF + XGBoost classifiers
python run_pipeline.py --module 5                   # open-set AUROC evaluation
streamlit run src/module6_dashboard.py              # launch dashboard
```

### Useful flags

| Flag | Module | Effect |
|---|---|---|
| `--reset` | 1 | Clear DB and re-ingest from scratch |
| `--no-llm` | 2 | Skip paraphrase/multilingual (no Ollama needed) |
| `--limit N` | 2 | Mutate only first N seeds (for testing) |
| `--min-cluster-size N` | 3 | HDBSCAN sensitivity (default 15) |
| `--held-out A B` | 4, 5 | Override which families to hold out for open-set eval |
| `--judge-sample N` | 5 | How many prompts the 8B model evaluates |

### Fastest end-to-end test (no Ollama needed)

```bash
python run_pipeline.py --module 1
python run_pipeline.py --module 2 --no-llm --limit 200
python run_pipeline.py --module 3
python run_pipeline.py --module 4
python run_pipeline.py --module 5
streamlit run src/module6_dashboard.py
```

Completes in ~30–40 minutes on any machine.

## Model strategy

| Task | Model | Why |
|---|---|---|
| Mutation generation | `llama3.2:3b` | Speed — runs thousands of mutations overnight |
| Judge evaluation | `llama3.1:8b-instruct-q4_K_M` | Quality — runs on small held-out set only |

## Key results (baseline run)

| Metric | Value |
|---|---|
| Corpus size after dedup | 1,085 prompts |
| Mutations generated | 400 (fast pass, 100 seeds) |
| HDBSCAN clusters | 8 |
| LogReg Macro F1 | 0.19 |
| XGBoost Macro F1 | 0.23 |
| **AUROC on held-out families** | **0.87** |
| Detection rate at hop=0 | 100% |
| Detection rate at hop=1 | 100% |
| Avg model compliance (8B judge) | 32% |

The low macro F1 reflects real class imbalance — `direct_harm` makes up ~60% of the corpus. The 0.87 AUROC shows the classifier generalizes to flag unseen attack families even without seeing them during training.

## Attack families

The system recognizes and classifies 10 canonical attack families:

| Family | Description |
|---|---|
| `direct_harm` | Explicit harmful requests with no obfuscation |
| `roleplay` | DAN/persona wrapping to bypass alignment |
| `encoding` | Base64, leetspeak, pig latin, unicode lookalikes, reversal |
| `indirect_injection` | Harmful instruction embedded in a document/email/code block |
| `social_engineering` | Authority framing, fake credentials, urgency |
| `persona_hijack` | Telling the model its "true self" has no restrictions |
| `token_smuggling` | Sentence completion, fill-in-the-blank, continuation tricks |
| `paraphrase` | Academically reworded harmful requests |
| `multilingual` | Non-English requests or translate-then-back-translate |
| `obfuscation` | Backward text, pig latin rewriting, stylistic disguise |

## After running Module 3

HDBSCAN auto-assigns family labels from the plurality `attack_type` within each cluster. You can override any label by editing `CLUSTER_LABEL_MAP` in [src/module3_cluster.py](src/module3_cluster.py) and re-running Module 3.

## Dashboard

```bash
streamlit run src/module6_dashboard.py
```

Features:
- Paste any prompt → predicted attack family + confidence score
- **Novel attack flagging** — confidence < 40% triggers a human-review alert
- Nearest known prompt in embedding space (cosine similarity)
- Mutation distance (hop count from original seed)
- Live interactive UMAP scatter colored by attack family
- Sidebar: corpus stats, family distribution, seed vs mutation counts
