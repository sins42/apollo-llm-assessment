import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")


def get_connection():
    # Opens a new connection to the database using the URL from .env.
    # Create a fresh connection per request rather than keeping one open
    # globally -- a single persistent connection would break under concurrent users.
    return psycopg2.connect(DB_URL)


def fetch_candidates(embedding: list[float], top_k: int = 15) -> list[dict]:
    conn = get_connection()

    # RealDictCursor makes each row come back as a dictionary like
    # {"id": "EX_001", "title": "Single-Leg Box Squat", ...}
    # instead of a plain tuple like ("EX_001", "Single-Leg Box Squat", ...)
    # which makes the data much easier to work with downstream
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """
        SELECT id, title, description, tags, body_part, difficulty,
                equipment, injury_focus, intensity,
                1 - (embedding <=> %s::vector) AS score
        FROM exercises
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (embedding, embedding, top_k),
    )

    results = [dict(row) for row in cur.fetchall()]

    cur.close()
    conn.close()

    return results