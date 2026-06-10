"""
SQLite store for the entire prompt corpus.
Schema:
  prompts   — raw + mutated prompts with metadata
  clusters  — cluster assignments from HDBSCAN
  evals     — evaluation results (judge scores, classifier predictions)
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "corpus.db"


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS prompts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            text        TEXT NOT NULL,
            source      TEXT,          -- jailbreakbench | advbench | harmbench | synthetic
            attack_type TEXT,          -- roleplay | encoding | multilingual | injection | direct | paraphrase
            language    TEXT DEFAULT 'en',
            parent_id   INTEGER REFERENCES prompts(id),
            mutation_op TEXT,          -- which mutator produced this
            embedding   BLOB,          -- serialized float32 numpy array
            fitness     REAL,          -- judge compliance score 0-1 (set by module7_evolve)
            generation  INTEGER DEFAULT 0,  -- evolutionary generation (0 = seed)
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS clusters (
            prompt_id   INTEGER PRIMARY KEY REFERENCES prompts(id),
            cluster_id  INTEGER,       -- -1 = noise in HDBSCAN
            family_label TEXT,         -- human-assigned: roleplay, encoding, social_engineering, etc.
            umap_x      REAL,
            umap_y      REAL
        );
        CREATE TABLE IF NOT EXISTS evals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id       INTEGER REFERENCES prompts(id),
            model           TEXT,
            response        TEXT,
            judge_verdict   TEXT,      -- REFUSAL | PARTIAL | COMPLIANCE
            judge_score     REAL,
            classifier_pred TEXT,      -- predicted attack family
            classifier_conf REAL,
            eval_method     TEXT,      -- qlora | llm_judge | rule_based
            created_at      TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_prompts_source ON prompts(source);
        CREATE INDEX IF NOT EXISTS idx_prompts_attack  ON prompts(attack_type);
        CREATE INDEX IF NOT EXISTS idx_clusters_id     ON clusters(cluster_id);
        """)


@contextmanager
def connect(path: Path = DB_PATH):
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def insert_prompt(text: str, source: str = None, attack_type: str = None,
                  language: str = "en", parent_id: int = None,
                  mutation_op: str = None, path: Path = DB_PATH) -> int:
    with connect(path) as con:
        cur = con.execute(
            "INSERT INTO prompts (text, source, attack_type, language, parent_id, mutation_op) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (text, source, attack_type, language, parent_id, mutation_op)
        )
        return cur.lastrowid


def bulk_insert_prompts(rows: list[dict], path: Path = DB_PATH) -> int:
    """rows: list of dicts with keys matching prompts columns. Returns count inserted."""
    with connect(path) as con:
        con.executemany(
            "INSERT OR IGNORE INTO prompts (text, source, attack_type, language, parent_id, mutation_op) "
            "VALUES (:text, :source, :attack_type, :language, :parent_id, :mutation_op)",
            rows
        )
        return con.total_changes


def get_prompts(source: str = None, limit: int = None, path: Path = DB_PATH) -> list[sqlite3.Row]:
    sql = "SELECT * FROM prompts"
    params = []
    if source:
        sql += " WHERE source = ?"
        params.append(source)
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with connect(path) as con:
        return con.execute(sql, params).fetchall()


def count_prompts(path: Path = DB_PATH) -> int:
    with connect(path) as con:
        return con.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]


def migrate_add_fitness_columns(path: Path = DB_PATH) -> None:
    """Add fitness and generation columns to existing DBs (safe no-op if already present)."""
    with connect(path) as con:
        existing = {row[1] for row in con.execute("PRAGMA table_info(prompts)")}
        if "fitness" not in existing:
            con.execute("ALTER TABLE prompts ADD COLUMN fitness REAL")
        if "generation" not in existing:
            con.execute("ALTER TABLE prompts ADD COLUMN generation INTEGER DEFAULT 0")
