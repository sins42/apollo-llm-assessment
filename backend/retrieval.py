from sentence_transformers import SentenceTransformer
from database import fetch_candidates
from config import EMBEDDING_MODEL

# Load the model once at module level rather than inside the function.
# Loading a model is expensive -- it reads weights from disk into memory.
# If loaded inside retrieve_candidates(), it would reload every time
# a user makes a request which would make the API very slow.
model = SentenceTransformer(EMBEDDING_MODEL)


def retrieve_candidates(query: str, top_k: int = 15) -> list[dict]:
    # Convert the user's text query into a 384-dimensional vector
    # using the same model used in seed.py to embed the exercises.
    # It's critical both sides use the same model
    embedding = model.encode(query).tolist()

    # Hand off to database.py which handles all SQL -- keeps this file
    # focused purely on the embedding logic
    return fetch_candidates(embedding, top_k)