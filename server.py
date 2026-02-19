from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import train_and_recommend
from recommender import refine_with_vibe, Recommendation
from scraper import LetterboxdScrapeError, scrape_letterboxd_films
from tmdb_client import TMDbClient

app = FastAPI(title="Movie Recommender API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production should be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    url: str
    count: int = 10
    debug: bool = False

class RecommendResponse(BaseModel):
    count: int
    recommendations: List[Recommendation]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(request: RecommendRequest):
    try:
        # 1. Scrape
        print(f"Scraping {request.url}...")
        user_df = scrape_letterboxd_films(request.url, debug=request.debug)
        
        # 2. Get API Key
        tmdb_key = os.environ.get("TMDB_API_KEY")
        tmdb_client = TMDbClient(api_key=tmdb_key) if tmdb_key else None
        
        # 3. Collaborative Filtering
        print("Training model...")
        artifacts = train_and_recommend(tmdb_client=tmdb_client)
        
        # 4. Refine with Vibe
        print("Refining recommendations...")
        recs = refine_with_vibe(
            candidates_df=artifacts.candidates,
            user_history_df=artifacts.user_history,
            tmdb_ids=artifacts.tmdb_ids,
            tmdb_client=tmdb_client,
            top_n=request.count,
            seen_titles=artifacts.seen_titles
        )
        
        return RecommendResponse(count=len(recs), recommendations=recs)

    except LetterboxdScrapeError as e:
        print(f"Scrape Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
