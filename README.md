# Jailbreak Genome Lab

A local, zero-cost research system for studying LLM robustness by evolving and classifying jailbreak prompts. Built as a portfolio project targeting AI Safety / Integrity roles.

## What it does

1. **Ingests** real jailbreak datasets (JailbreakBench, LibrAI/do-not-answer, built-in seeds)
2. **Mutates** prompts using 5 strategies: encoding, roleplay, indirect injection, paraphrase, multilingual
3. **Clusters** the full corpus with HDBSCAN + UMAP to discover attack family structure
4. **Classifies** attack families using TF-IDF + LogReg/XGBoost baseline, then QLoRA fine-tuned Llama 3.2 3B
5. **Evaluates** open-set recognition — can the classifier flag *unseen* attack families as anomalous?
6. **Visualizes** everything in a live Streamlit dashboard

Key result: **0.87 AUROC** on held-out attack families (encoding + multilingual) using only a TF-IDF baseline, with QLoRA stage available for further improvement.

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
  classifier/
    qlora_adapter/   ← populated after Module 4 QLoRA stage
  results/
    eval_report.json

src/
  db.py              ← SQLite schema + helpers
  llm_runner.py      ← Ollama HTTP client (llama3.2:3b + llama3.1:8b)
  module1_ingest.py  ← Dataset download + dedup
  module2_mutate.py  ← 5 mutation strategies
  module3_cluster.py ← HDBSCAN + UMAP
  module4_classify.py← TF-IDF baseline + QLoRA SFT
  module5_evaluate.py← Open-set AUROC + mutation distance curve
  module6_dashboard.py← Streamlit UI

run_pipeline.py      ← Master orchestrator
requirements.txt
```

## Setup

### 1. Python dependencies

```bash
pip install -r requirements.txt

# QLoRA fine-tuning (GPU recommended, CPU works but is slow)
pip install torch transformers peft bitsandbytes trl accelerate
```

### 2. Ollama (local LLM runner)

Download from [ollama.com](https://ollama.com) and pull the two models:

```bash
ollama pull llama3.2:3b        # mutation engine — fast, high volume
ollama pull llama3.1:8b-instruct-q4_K_M  # judge evaluator — quality
```

Ollama must be running (`ollama serve`) before using Module 2 LLM mutations or Module 5 judge evaluation.

## Running the pipeline

```bash
# Full pipeline (sequential)
python run_pipeline.py

# Or module by module:
python run_pipeline.py --module 1                   # ingest ~1k prompts
python run_pipeline.py --module 2 --no-llm          # fast mutations (no Ollama needed)
python run_pipeline.py --module 2                   # add LLM mutations (needs Ollama)
python run_pipeline.py --module 3                   # cluster + UMAP
python run_pipeline.py --module 4 --stage baseline  # TF-IDF + XGBoost
python run_pipeline.py --module 4 --stage qlora     # QLoRA fine-tune (GPU recommended)
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
| `--stage baseline\|qlora\|both` | 4 | Which classifier to train |
| `--held-out A B` | 4, 5 | Override which families to hold out |
| `--judge-sample N` | 5 | How many prompts the 8B model evaluates |

## Model strategy

| Task | Model | Why |
|---|---|---|
| Mutation generation | `llama3.2:3b` | Speed — runs thousands of mutations overnight |
| Judge evaluation | `llama3.1:8b-instruct-q4_K_M` | Quality — runs on small held-out set only |
| QLoRA classifier base | `llama3.2:3b` | Fits in 8GB VRAM with 4-bit quantization |

## Key results (baseline run)

| Metric | Value |
|---|---|
| Corpus size after dedup | 1,085 prompts |
| Mutations generated | 400 (fast pass, 100 seeds) |
| HDBSCAN clusters | 8 |
| Classifier accuracy | 84% |
| Macro F1 (seen families) | 0.29 |
| **AUROC on held-out families** | **0.87** |
| Detection rate at hop=0 | 100% |
| Detection rate at hop=1 | 100% |
| Avg model compliance (8B judge) | 32% |

The low macro F1 reflects real class imbalance — `direct_harm` makes up ~60% of the corpus. Encoding-specific subtypes and roleplay/indirect_injection clusters perform well (F1 > 0.85 each). The 0.87 AUROC shows the classifier generalizes to flag unseen attack families even without seeing them during training.

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

## QLoRA fine-tuning details

- Base model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- LoRA rank: 16, alpha: 32, targets: q/k/v/o projections
- Training task: predict attack family name from prompt text
- Tries `unsloth` first (faster), falls back to `peft + bitsandbytes`
- Adapter saved to `data/classifier/qlora_adapter/`

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
