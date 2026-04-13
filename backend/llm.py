"""
llm.py — Gemini Flash integration for PriceIQ Mini
Builds a structured prompt from ML outputs and calls Gemini API.
The LLM never predicts price — it only explains the ML model's output.
"""

import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=_API_KEY)
_gemini = genai.GenerativeModel("gemini-1.5-flash")


def build_prompt(catalog_content: str, predicted_price: float, top_signals: dict) -> str:
    """
    Build the structured XML prompt that grounds Gemini in ML model reasoning.
    top_signals are the 40-feature values extracted during inference.
    """
    cat       = top_signals.get("category", "general")
    count     = top_signals.get("pack_count", 1)
    premium   = top_signals.get("premium_score", 0)
    bulk      = top_signals.get("bulk_score", 0)
    brand     = top_signals.get("brand_score", 0)
    diet      = top_signals.get("has_special_diet", 0)
    quality   = top_signals.get("quality_indicator", 0)
    tier      = top_signals.get("price_tier", 0)

    tier_label = (
        "premium tier"   if tier > 1
        else "budget tier" if tier < -1
        else "mid-range tier"
    )

    prompt = f"""You are a product pricing analyst for an e-commerce platform.
You have been given the output from a trained multimodal ML model (CLIP ViT-L/14 + 40 tabular features).
Your job is to explain WHY this product costs this much based on the ML model's feature signals.
Do NOT invent reasons outside what the signals tell you. Be grounded and honest.

<product>
  catalog_content: {catalog_content[:500]}
</product>

<ml_model_output>
  predicted_price: ${predicted_price:.2f}
  model_signals:
    - category_detected: {cat}
    - pack_count: {count:.0f} units
    - premium_keywords_score: {premium:.0f}
    - bulk_indicator_score: {bulk:.0f}
    - brand_signal_score: {brand:.0f}
    - special_diet_flag: {"yes" if diet else "no"}
    - quality_indicator: {quality:.1f}/5
    - price_tier: {tier_label}
</ml_model_output>

Based ONLY on the above signals, respond in valid JSON with exactly these keys:
{{
  "price_explanation": "2-3 sentences explaining why this product costs ${predicted_price:.2f} using the model signals above",
  "key_features": ["feature1", "feature2", "feature3"],
  "target_customers": "one sentence describing who buys this",
  "value_verdict": "Good value" or "Fair price" or "Premium priced",
  "value_reasoning": "one sentence justifying the verdict"
}}

Return ONLY the JSON object. No markdown, no preamble, no explanation outside the JSON."""

    return prompt


def call_gemini(prompt: str) -> dict:
    """
    Call Gemini Flash and parse JSON response.
    Returns parsed dict or a safe fallback on error.
    """
    try:
        response = _gemini.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=512,
            ),
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except Exception as e:
        # Return a safe fallback so the API never crashes on LLM errors
        return {
            "price_explanation": "Price determined by the ML model based on product features, pack size, and quality signals.",
            "key_features": ["multimodal analysis", "pack quantity", "product category"],
            "target_customers": "General consumers looking for this product category.",
            "value_verdict": "Fair price",
            "value_reasoning": f"Price reflects product characteristics detected by the model. (LLM error: {str(e)[:80]})",
        }


def get_explanation(catalog_content: str, predicted_price: float, top_signals: dict) -> dict:
    """
    Main entry point: build prompt → call Gemini → return parsed JSON.
    """
    prompt = build_prompt(catalog_content, predicted_price, top_signals)
    return call_gemini(prompt)