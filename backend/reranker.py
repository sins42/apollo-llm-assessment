import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize the Groq client once at module level, same reasoning as the
# sentence-transformers model in retrieval.py -- no need to recreate it per request
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    # Build a numbered list of candidates to include in the prompt.
    # Include the most descriptive fields so the LLM has enough context
    # to make a meaningful relevance judgment -- just sending titles wouldn't
    # give it enough to work with.
    candidate_lines = "\n".join(
        f"{i+1}. [{c['id']}] {c['title']} - {c['description']} "
        f"(body: {c['body_part']}, difficulty: {c['difficulty']}, "
        f"equipment: {c['equipment']}, injury focus: {c['injury_focus']})"
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are an expert sports coaching assistant. A user submitted this query:
"{query}"

Here are candidate exercises retrieved from a database:
{candidate_lines}

Return the top {top_n} most relevant exercises as a JSON array with this exact format:
[{{"id": "EX_001", "reason": "brief reason why this matches the query"}}, ...]

Return only valid JSON, no extra text."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        # Low temperature makes the model more deterministic and consistent.
        # Don't want creative answers here, only want reliable JSON output.
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    # Parse the JSON the LLM returned into a Python list
    ranked_ids = json.loads(raw)

    # Build a lookup map so we can find each exercise by id in O(1)
    # rather than looping through the candidates list for every result
    candidate_map = {c["id"]: c for c in candidates}

    results = []
    for item in ranked_ids:
        if item["id"] in candidate_map:
            exercise = candidate_map[item["id"]]
            # Attach the LLM's reason to the exercise so the frontend can display it
            exercise["reason"] = item["reason"]
            results.append(exercise)

    return results