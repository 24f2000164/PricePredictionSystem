"""
main.py — FastAPI backend for PriceIQ Mini
Exposes POST /api/predict and GET /api/health.
Model is loaded once at startup via pipeline.py.
Run: uvicorn main:app --reload --port 8000
"""

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from pipeline import predict_price
from llm import get_explanation
from cache import make_key, get_cache, set_cache

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PriceIQ Mini",
    description="Multimodal price prediction + LLM explanation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    catalog_content: str
    image_link: str          # URL string (validated loosely so http/https both work)


class PredictResponse(BaseModel):
    predicted_price:  float
    price_explanation: str
    key_features:     list[str]
    target_customers: str
    value_verdict:    str
    value_reasoning:  str
    top_signals:      dict
    cached:           bool


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model": "CLIP ViT-L/14 + EnhancedPriceHead", "llm": "Gemini Flash"}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    # 1. Check cache
    cache_key = make_key(req.catalog_content, req.image_link)
    cached    = get_cache(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    # 2. Fetch image bytes
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            img_resp = await client.get(
                req.image_link,
                headers={"User-Agent": "PriceIQ/1.0"},
                follow_redirects=True,
            )
            img_resp.raise_for_status()
            image_bytes = img_resp.content
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not fetch image: {e}")

    # 3. ML inference
    try:
        ml_result = predict_price(req.catalog_content, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    predicted_price = ml_result["predicted_price"]
    top_signals     = ml_result["top_signals"]

    # 4. LLM explanation (Gemini)
    try:
        llm_result = get_explanation(req.catalog_content, predicted_price, top_signals)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM explanation failed: {e}")

    # 5. Build response
    response = {
        "predicted_price":   predicted_price,
        "price_explanation": llm_result.get("price_explanation", ""),
        "key_features":      llm_result.get("key_features", []),
        "target_customers":  llm_result.get("target_customers", ""),
        "value_verdict":     llm_result.get("value_verdict", "Fair price"),
        "value_reasoning":   llm_result.get("value_reasoning", ""),
        "top_signals":       top_signals,
        "cached":            False,
    }

    # 6. Cache and return
    set_cache(cache_key, response)
    return response