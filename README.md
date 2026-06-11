# Jailbreak Genome Lab

A local, zero-cost research system for studying LLM robustness by evolving and classifying jailbreak prompts. Built as a portfolio project targeting AI Safety / Integrity roles.

![Jailbreak Genome Lab Dashboard](docs/dashboard.png)

## What it does

1. **Ingests** real jailbreak datasets (JailbreakBench, LibrAI/do-not-answer, built-in seeds)
2. **Mutates** prompts using 5 strategies: encoding, roleplay, indirect injection, paraphrase, multilingual
3. **Clusters** the full corpus with HDBSCAN + UMAP to discover attack family structure
4. **Classifies** attack families using TF-IDF + Logistic Regression + XGBoost
5. **Evaluates** open-set recognition — can the classifier flag *unseen* attack families as anomalous?
6. **Visualizes** everything in a live Streamlit dashboard
7. **Evolves** a next-generation corpus via fitness-guided selection — score each prompt, select the top performers, mutate them, repeat

Key results: embedding-based k-NN beats a fairly-trained TF-IDF baseline on novel attack technique detection in **3 of 4 holdout sets** (up to **0.82 vs 0.67 AUROC** on encoding+multilingual) — and loses exactly where embedding distance *should* fail (paraphrase, where novel prompts are semantically identical to known ones). See [Limitations](#limitations) for what these numbers do and don't show.

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
    evolution_report.json   ← per-generation fitness stats

src/
  db.py               ← SQLite schema + helpers
  llm_runner.py       ← Ollama HTTP client (llama3.2:3b + llama3.1:8b)
  module1_ingest.py   ← Dataset download + dedup
  module2_mutate.py   ← 5 mutation strategies
  module3_cluster.py  ← HDBSCAN + UMAP
  module4_classify.py ← TF-IDF + LogReg + XGBoost
  module5_evaluate.py ← Open-set AUROC + mutation distance curve
  module6_dashboard.py← Streamlit UI
  module7_evolve.py   ← Fitness-guided evolutionary search

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
python src/module7_evolve.py                        # run evolutionary search (needs Ollama)
python src/module7_evolve.py --generations 3 --eval-sample 30 --no-llm  # fast test
```

### Useful flags

| Flag | Module | Effect |
|---|---|---|
| `--reset` | 1 | Clear DB and re-ingest from scratch |
| `--no-llm` | 2, 7 | Skip paraphrase/multilingual (no Ollama needed) |
| `--limit N` | 2 | Mutate only first N seeds (for testing) |
| `--min-cluster-size N` | 3 | HDBSCAN sensitivity (default 15) |
| `--held-out A B` | 4, 5 | Override which families to hold out for open-set eval |
| `--judge-sample N` | 5 | How many prompts the 8B model evaluates |
| `--generations N` | 7 | Number of evolutionary generations (default 3) |
| `--eval-sample N` | 7 | Prompts scored per generation (default 30) |
| `--selection-rate F` | 7 | Top fraction selected as parents (default 0.3) |
| `--start-gen N` | 7 | Resume evolution from generation N |

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

### Run the evolutionary loop

Requires Ollama running with both models pulled.

```bash
# 3 generations, 30 prompts scored per gen, top 30% selected as parents
python src/module7_evolve.py --generations 3 --eval-sample 30

# Full run (slow — each generation scores 100 prompts with the 8B judge)
python src/module7_evolve.py --generations 5 --eval-sample 100

# Resume if interrupted (picks up from generation 2)
python src/module7_evolve.py --start-gen 2 --generations 5
```

Results are saved to `data/results/evolution_report.json` with per-generation fitness stats.

## Model strategy

| Task | Model | Why |
|---|---|---|
| Mutation generation (M2, M7) | `llama3.2:3b` | Speed — runs thousands of mutations overnight |
| Judge evaluation (M5, M7) | `llama3.1:8b-instruct-q4_K_M` | Quality — fitness scoring and held-out eval |

## Key results

Two runs were compared: **no-LLM mutations** (rule-based only) vs **LLM mutations** (llama3.2:3b paraphrase + multilingual).

### Corpus (after LLM run)
| Metric | Value |
|---|---|
| Total prompts | 5,424 |
| Seeds | 1,085 |
| Attack families detected | 13 |
| Largest family | `direct_harm` (1,770) |

![Corpus Growth](data/results/plots/latest_corpus_growth.png)

### Novelty Detection — TF-IDF vs Embedding k-NN (fair comparison)

Both methods are trained on the **same holdout-excluded split**, then asked to flag the held-out families as novel. (An earlier version of this comparison reused a TF-IDF model that had seen the held-out families during training, which pushed its AUROC below chance — 0.32 — and inflated the gap. These are the corrected numbers.)

| Held-out set | TF-IDF max-conf | Embedding k-NN (e5-base) | Winner |
|---|---|---|---|
| encoding + multilingual | 0.675 | **0.816** | k-NN |
| roleplay + persona_hijack | 0.582 | **0.741** | k-NN |
| indirect_injection | 0.676 | **0.752** | k-NN |
| paraphrase + token_smuggling | **0.540** | 0.442 | TF-IDF |

k-NN wins where novelty has a surface/style signature (encodings, other languages, persona framing) and loses on paraphrase — by construction, paraphrased prompts are semantically identical to known ones, so embedding distance cannot see them as novel. The failure mode is as informative as the wins: novelty detectors need both lexical and semantic signals.

![AUROC History](data/results/plots/latest_auroc_history.png)

### Classifier F1 over runs

![F1 History](data/results/plots/latest_f1_history.png)

### Open-Set AUROC by held-out family (LLM run)
| Held-out family | AUROC |
|---|---|
| `indirect_injection` | 0.676 |
| `encoding + multilingual` | 0.675 |
| `roleplay + persona_hijack` | 0.582 |
| `paraphrase + token_smuggling` | 0.540 |

### Effect of LLM mutations on evasion
| Held-out family | No-LLM AUROC | LLM AUROC | Change |
|---|---|---|---|
| encoding+multilingual | 0.737 | 0.675 | −0.062 |
| roleplay+persona_hijack | 0.694 | 0.582 | −0.112 |
| indirect_injection | 0.746 | 0.676 | −0.070 |

**LLM mutations lower AUROC across all families** — consistent with LLM rewriting producing more evasive variants. Caveat: the two runs differ in corpus composition *and* clustering (69 vs 97 clusters), so part of the drop may be label shift rather than evasiveness; a controlled version of this experiment would hold the clustering fixed and swap only the mutation set.

### Mutation distance curve

![Mutation Distance Curve](data/results/plots/latest_mutation_distance_curve.png)

| Hop depth | Detection rate | Novel rate |
|---|---|---|
| 0 (original seeds) | 5.5% | 94.5% |
| 1 (direct mutations) | 72.5% | 27.5% |

The classifier is far more confident on mutations than on original seeds. This is mostly a class-prior effect — the training corpus is 80% mutations, so the model is fitted to mutation styles — and it means the dashboard's 0.40 novelty threshold would flag nearly every original-style jailbreak for human review. The curve currently only covers hops 0–1; deeper lineage requires running Module 7 evolution.

## Limitations

Honest scoping of what the numbers above show:

1. **Labels are attack *techniques*, not harm domains.** ~80% of the corpus is mutations, labeled by the mutation operator that produced them (87% of seeds share one harm label, `direct_harm`). "Detect a novel family" therefore means "detect the output of a held-out transformation technique" — a real and useful task (new obfuscation styles appear in the wild), but not the same as detecting novel harmful *intent*. The paraphrase holdout is nearly self-defeating by construction, which is why every method scores at or below chance on it.
2. **Per-family classification is weak** (0.41–0.43 macro F1). Partly label noise from the technique/family conflation above; the AUROC results are the robust finding, not the classifier itself.
3. **The LLM-evasion comparison is confounded** by clustering changes between runs (see caveat above).
4. **Judge scores** (when enabled) come from a 3B/8B local model; parse failures are tracked and excluded from averages rather than silently scored as 0.5.

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
