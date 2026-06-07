"""
Module 4: Two-Stage Classifier Training
Stage 1 — TF-IDF + Logistic Regression + XGBoost (baseline)
Stage 2 — QLoRA fine-tuned Llama 3.2 3B via unsloth/peft (SFT on attack family labels)
"""
import sys
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from db import connect
from llm_runner import MUTATION_MODEL

ARTIFACTS_DIR  = Path(__file__).parent.parent / "data" / "artifacts"
CLASSIFIER_DIR = Path(__file__).parent.parent / "data" / "classifier"
ADAPTER_PATH   = CLASSIFIER_DIR / "qlora_adapter"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labeled_data(held_out_families: list[str] = None) -> tuple[list, list, list, list]:
    """
    Returns (X_train, y_train, X_test, y_test).
    Label priority: manual cluster family_label > mutation_op > attack_type from prompts.
    If held_out_families given, those are excluded from training (open-set eval).
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

    if held_out_families:
        train_x, train_y, test_x, test_y = [], [], [], []
        for t, l in zip(texts, labels):
            if l in held_out_families:
                test_x.append(t); test_y.append(l)
            else:
                train_x.append(t); train_y.append(l)
        return train_x, train_y, test_x, test_y

    # Standard 80/20 split
    from sklearn.model_selection import train_test_split
    from collections import Counter
    # stratify only when every class has >=2 samples
    counts = Counter(labels)
    can_stratify = all(v >= 2 for v in counts.values())
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=42,
        stratify=labels if can_stratify else None,
    )
    return X_tr, y_tr, X_te, y_te


# ── Stage 1: Baseline ─────────────────────────────────────────────────────────

def train_baseline(X_train, y_train, X_test, y_test) -> dict:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
        print("  XGBoost not installed, skipping. pip install xgboost")

    results = {}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Logistic Regression
    print("\n--- Stage 1a: TF-IDF + Logistic Regression ---")
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0)),
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
    if HAS_XGB:
        print("\n--- Stage 1b: TF-IDF + XGBoost ---")
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)

        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf  = tfidf.transform(X_test)

        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="mlogloss",
                             verbosity=0, tree_method="hist")
        xgb.fit(X_train_tfidf, y_train_enc)
        y_pred_xgb = le.inverse_transform(xgb.predict(X_test_tfidf))
        f1_xgb = f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)
        print(classification_report(y_test, y_pred_xgb, zero_division=0))
        print(f"Macro F1: {f1_xgb:.4f}")
        results["xgboost_f1"] = f1_xgb

        with open(ARTIFACTS_DIR / "baseline_xgb.pkl", "wb") as f:
            pickle.dump((tfidf, xgb, le), f)

    return results


# ── Stage 2: QLoRA fine-tuned classifier ─────────────────────────────────────

CLASSIFICATION_PROMPT = """\
Classify the following jailbreak prompt into one of these attack families:
roleplay, encoding, social_engineering, persona_hijack, token_smuggling,
indirect_injection, multilingual, paraphrase, direct_harm, obfuscation

Output ONLY the family name, nothing else.

Prompt: {text}

Family:"""


def prepare_sft_dataset(X_train: list[str], y_train: list[str]) -> Path:
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for text, label in zip(X_train, y_train):
        full = CLASSIFICATION_PROMPT.format(text=text[:400]) + " " + label
        samples.append({"text": full, "label": label})
    path = CLASSIFIER_DIR / "sft_dataset.json"
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SFT dataset saved: {path} ({len(samples)} samples)")
    return path


def train_qlora(X_train: list[str], y_train: list[str],
                base_model: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                epochs: int = 3, batch_size: int = 2) -> None:
    dataset_path = prepare_sft_dataset(X_train, y_train)

    # Try unsloth first, fall back to peft
    try:
        _train_with_unsloth(dataset_path, base_model, epochs, batch_size)
    except ImportError:
        print("unsloth not available — using peft + bitsandbytes directly.")
        _train_with_peft(dataset_path, base_model, epochs, batch_size)


def _train_with_unsloth(dataset_path: Path, base_model: str, epochs: int, batch_size: int) -> None:
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=512, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32, lora_dropout=0.05, bias="none", use_gradient_checkpointing=True,
    )

    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset = Dataset.from_list(samples)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=512,
        args=TrainingArguments(
            output_dir=str(ADAPTER_PATH), num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=4,
            learning_rate=2e-4, fp16=True, logging_steps=10,
            save_strategy="epoch", report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(str(ADAPTER_PATH))
    tokenizer.save_pretrained(str(ADAPTER_PATH))
    print(f"Unsloth QLoRA adapter saved to {ADAPTER_PATH}")


def _train_with_peft(dataset_path: Path, base_model: str, epochs: int, batch_size: int) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb,
                                                   device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=Dataset.from_list(samples),
        dataset_text_field="text", max_seq_length=512,
        args=TrainingArguments(
            output_dir=str(ADAPTER_PATH), num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=4,
            learning_rate=2e-4, fp16=True, logging_steps=10,
            save_strategy="epoch", report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(str(ADAPTER_PATH))
    tokenizer.save_pretrained(str(ADAPTER_PATH))
    print(f"PEFT QLoRA adapter saved to {ADAPTER_PATH}")


# ── QLoRA inference ───────────────────────────────────────────────────────────

class QLoRAFamilyClassifier:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        cfg_path = ADAPTER_PATH / "adapter_config.json"
        base = json.loads(cfg_path.read_text())["base_model_name_or_path"]
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
        base_model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
                                                            device_map="auto")
        self.model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def predict(self, text: str) -> tuple[str, float]:
        import torch
        inp = CLASSIFICATION_PROMPT.format(text=text[:400])
        tokens = self.tokenizer(inp, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**tokens, max_new_tokens=8, temperature=0.1,
                                       do_sample=False,
                                       pad_token_id=self.tokenizer.eos_token_id)
        generated = self.tokenizer.decode(
            out[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()

        from module3_cluster import KNOWN_FAMILIES
        for fam in KNOWN_FAMILIES:
            if fam in generated:
                return fam, 0.9
        return "unknown", 0.1


# ── Main ──────────────────────────────────────────────────────────────────────

def run(stage: str = "both", held_out: list[str] = None, epochs: int = 3) -> dict:
    print("=== Module 4: Classifier Training ===\n")

    X_train, y_train, X_test, y_test = load_labeled_data(held_out_families=held_out)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    if not X_train:
        print("No labeled data found. Run Module 3 first and assign CLUSTER_LABEL_MAP.")
        return {}

    results = {}

    if stage in ("baseline", "both"):
        baseline_results = train_baseline(X_train, y_train, X_test, y_test)
        results.update(baseline_results)

    if stage in ("qlora", "both"):
        print("\n--- Stage 2: QLoRA Fine-tuning ---")
        train_qlora(X_train, y_train, epochs=epochs)
        results["qlora_trained"] = True

    print("\nModule 4: Done.")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["baseline", "qlora", "both"], default="both")
    p.add_argument("--held-out", nargs="+", default=None,
                   help="Attack families to hold out for open-set eval")
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    run(stage=args.stage, held_out=args.held_out, epochs=args.epochs)
