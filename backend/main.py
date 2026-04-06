import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from retrieval import retrieve_candidates
from reranker import rerank

app = FastAPI()

# CORS allows the frontend (running in a browser) to make requests to this API.
# Without this, the browser blocks requests coming from a different origin.
# allow_origins=["*"] is fine for local development but in production must 
# restrict this to actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend folder using an absolute path
# Accessible at http://localhost:8000/static/index.html
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend")), name="static")

# Pydantic model defines the shape of the request body.
# FastAPI uses this to automatically validate incoming requests 
class QueryRequest(BaseModel):
    query: str


@app.post("/recommend")
def recommend(request: QueryRequest):
    # Guard against empty queries that would produce meaningless results
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1 -- retrieval: embed the query and find the 15 closest exercises
    # using vector similarity search. No LLM involved here.
    candidates = retrieve_candidates(request.query)

    # Step 2 -- re-ranking: send those candidates to the LLM which reads
    # the query and each exercise description and returns the top 5 by relevance.
    results = rerank(request.query, candidates)

    return {"results": results}