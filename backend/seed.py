import os
import csv
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from config import EMBEDDING_MODEL

# Load environment variables from .env so we don't hardcode credentials
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

# Paths are relative to this file's location 
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "exercises.csv")
INDEX_SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "seed_index.sql")


def build_embedding_text(row):
    # Combine the most descriptive fields into a single string for embedding.
    # The more context the model has, the better the semantic search will be.
    # Exclude id and difficulty since they don't add semantic meaning.
    return (
        f"{row['title']}. {row['description']}. "
        f"Tags: {row['tags']}. Body part: {row['body_part']}. "
        f"Injury focus: {row['injury_focus']}. Equipment: {row['equipment']}."
    )


def main():
    print("Loading sentence-transformers model...")
    # all-MiniLM-L6-v2 is lightweight, fast on CPU, and outputs 384-dim vectors
    # which matches the vector(384) column defined in schema.sql
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Reading CSV...")
    with open(CSV_PATH, encoding="utf-8") as f:
        # Filter out any rows missing an id in case of blank trailing rows in the CSV
        rows = [r for r in csv.DictReader(f) if r.get("id", "").strip()]

    print(f"Generating embeddings for {len(rows)} exercises...")
    texts = [build_embedding_text(r) for r in rows]

    # encode() returns a numpy array of shape (num_rows, 384).
    # show_progress_bar gives visual feedback since this can take a moment on CPU
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Connecting to database...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    print("Inserting exercises...")
    for row, embedding in zip(rows, embeddings):
        cur.execute(
            """
            INSERT INTO exercises
                (id, title, description, tags, body_part, difficulty,
                equipment, injury_focus, intensity, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                row["id"], row["title"], row["description"], row["tags"],
                row["body_part"], row["difficulty"], row["equipment"],
                row["injury_focus"], row["intensity"],
                # psycopg2 needs the numpy array as a plain Python list
                embedding.tolist(),
            ),
        )

    # Commit all inserts as a single transaction -- if anything fails,
    # nothing gets written to the DB, keeping it in a clean state
    conn.commit()

    print("Building vector index...")
    with open(INDEX_SQL_PATH) as f:
        cur.execute(f.read())
    conn.commit()

    cur.close()
    conn.close()
    print("Done. Database is ready.")


if __name__ == "__main__":
    main()