"""
Loads medicines.csv, generates local embeddings, and upserts everything into
the Supabase `medicines` table (see supabase_setup.sql for the schema).

Run this whenever you add/update medicines:
    python load_data.py
"""

import csv
import logging
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("load_data")

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # Supabase connection string
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add your Supabase connection string to .env")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "medicines.csv"

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
DB_PAGE_SIZE = int(os.getenv("DB_PAGE_SIZE", "100"))  # rows per bulk upsert round trip


def derive_category(use_case: str) -> str:
    """First listed use-case becomes the coarse filterable category,
    e.g. "Fever, Headache" -> "Fever"."""
    if not use_case:
        return ""
    return use_case.split(",")[0].strip()


def derive_in_stock(stock: str) -> bool:
    return (stock or "").strip().lower() == "yes"


def row_to_text(row: dict) -> str:
    return (
        f"Medicine: {row['Medicine_Name']} ({row['Strength']}). "
        f"Used for: {row['Use_Case']}. "
        f"Alternatives: {row['Alternative']}. "
        f"In stock: {row['Stock']}. "
        f"Dosage: {row['Dosage_Instruction']}."
    )


def load_rows() -> list[dict]:
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required_columns = {
        "Medicine_ID", "Medicine_Name", "Strength", "Use_Case",
        "Alternative", "Stock", "Dosage_Instruction",
    }
    if rows:
        missing = required_columns - set(rows[0].keys())
        if missing:
            raise RuntimeError(f"medicines.csv is missing required columns: {missing}")

    return rows


def embed_texts(texts: list[str]) -> list[list[float]]:
    logger.info(
        "Embedding %d rows in batches of %d...", len(texts), EMBED_BATCH_SIZE
    )
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return [e.tolist() for e in embeddings]


def upsert_rows(rows: list[dict], texts: list[str], embeddings: list[list[float]]) -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()

        upsert_sql = """
            insert into medicines
                (medicine_id, medicine_name, strength, use_case, alternative,
                 stock, dosage_instruction, category, in_stock, content, embedding)
            values %s
            on conflict (medicine_id) do update set
                medicine_name      = excluded.medicine_name,
                strength           = excluded.strength,
                use_case           = excluded.use_case,
                alternative        = excluded.alternative,
                stock              = excluded.stock,
                dosage_instruction = excluded.dosage_instruction,
                category           = excluded.category,
                in_stock           = excluded.in_stock,
                content            = excluded.content,
                embedding          = excluded.embedding;
        """

        values = [
            (
                row["Medicine_ID"],
                row["Medicine_Name"],
                row["Strength"],
                row["Use_Case"],
                row["Alternative"],
                row["Stock"],
                row["Dosage_Instruction"],
                derive_category(row["Use_Case"]),
                derive_in_stock(row["Stock"]),
                text,
                emb,
            )
            for row, text, emb in zip(rows, texts, embeddings)
        ]

        logger.info(
            "Upserting %d rows in batches of %d...", len(values), DB_PAGE_SIZE
        )
        execute_values(cur, upsert_sql, values, page_size=DB_PAGE_SIZE)

        conn.commit()
        cur.close()
    finally:
        conn.close()


def main():
    rows = load_rows()
    if not rows:
        logger.warning("No rows found in medicines.csv - nothing to load.")
        return

    texts = [row_to_text(r) for r in rows]
    embeddings = embed_texts(texts)
    upsert_rows(rows, texts, embeddings)

    logger.info("Done - upserted %d medicines into Supabase.", len(rows))


if __name__ == "__main__":
    main()