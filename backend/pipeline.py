"""
pipeline.py — ML inference engine for PriceIQ Mini
Loads best_model.pt once at startup and exposes predict_price().
All model classes and feature functions are copied exactly from
MLPrice_improved_v2.ipynb to guarantee identical behaviour.
"""

import re, io, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# ── Install deps if missing ───────────────────────────────────────────────────
import subprocess, sys
for pkg, imp in [("open_clip_torch", "open_clip"), ("peft", "peft")]:
    try:
        __import__(imp)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

from peft import LoraConfig, get_peft_model
import open_clip

# ── Price normalisation stats ─────────────────────────────────────────────────
_STATS_PATH = os.path.join(os.path.dirname(__file__), "models", "price_stats.json")
with open(_STATS_PATH) as f:
    _stats = json.load(f)

LOG_MEAN: float = _stats["log_mean"]
LOG_STD:  float = _stats["log_std"]

# =============================================================================
# FEATURE ENGINEERING  (copied exactly from notebook cell_features_extract)
# =============================================================================

def extract_comprehensive_features(text: str) -> dict:
    """Extract 40 price-relevant features from catalog text."""
    text_lower = text.lower()

    # ── Physical measurements ─────────────────────────────────────────────
    weight_match = re.search(
        r"(\d+\.?\d*)\s*(oz|ounce|lb|pound|g|gram|kg|kilogram)", text_lower
    )
    weight = 0.0
    if weight_match:
        val, unit = float(weight_match.group(1)), weight_match.group(2)
        if "lb" in unit or "pound" in unit:
            weight = val * 16
        elif "kg" in unit or "kilogram" in unit:
            weight = val * 35.274
        elif "g" in unit and "kg" not in unit:
            weight = val * 0.035274
        else:
            weight = val

    volume_match = re.search(
        r"(\d+\.?\d*)\s*(ml|l|liter|litre|fl\s*oz|gallon|cup|quart)", text_lower
    )
    volume = 0.0
    if volume_match:
        val, unit = float(volume_match.group(1)), volume_match.group(2)
        if "gallon" in unit:
            volume = val * 3785.41
        elif "l" in unit and "ml" not in unit:
            volume = val * 1000
        elif "fl" in unit or "oz" in unit:
            volume = val * 29.5735
        elif "cup" in unit:
            volume = val * 236.588
        elif "quart" in unit:
            volume = val * 946.353
        else:
            volume = val

    count_match = re.search(
        r"pack\s*of\s*(\d+)|(\d+)\s*(pack|count|ct|piece|pc|box|case|bottle|jar)",
        text_lower,
    )
    count = 1.0
    if count_match:
        count = float(
            count_match.group(1) if count_match.group(1) else count_match.group(2)
        )

    value_match = re.search(r"value:\s*(\d+\.?\d*)", text_lower)
    extracted_value = float(value_match.group(1)) if value_match else 0.0

    # ── Brand & quality signals ───────────────────────────────────────────
    words = text[:100].split()
    brand_score = sum(1 for w in words[:5] if w and len(w) > 2 and w[0].isupper())

    premium_keywords = [
        "premium","organic","natural","gourmet","artisan","imported",
        "professional","certified","pure","luxury","deluxe","supreme",
        "ultra","pro","elite","signature","reserve",
    ]
    premium_score = sum(1 for k in premium_keywords if k in text_lower)

    budget_keywords = ["value","basic","economy","budget","essential","cheap","affordable"]
    budget_score = sum(1 for k in budget_keywords if k in text_lower)

    # ── Category detection ────────────────────────────────────────────────
    categories = {
        "food":        ["food","snack","drink","beverage","meal","chocolate","coffee","tea",
                        "sauce","jam","cookie","beans","cereal","soup","candy"],
        "electronics": ["electronic","cable","charger","adapter","usb","hdmi","phone"],
        "beauty":      ["beauty","skin","hair","cosmetic","makeup","cream","lotion","perfume","foundation"],
        "home":        ["home","kitchen","clean","storage","organize","towel","furniture"],
        "health":      ["vitamin","supplement","health","protein","probiotic","medicine","tea","organic"],
        "spices":      ["spice","seasoning","salt","pepper","basil","sage","powder"],
    }
    cat_scores = {
        cat: sum(1 for k in kws if k in text_lower)
        for cat, kws in categories.items()
    }
    dominant_cat = (
        max(cat_scores, key=cat_scores.get) if max(cat_scores.values()) > 0 else "other"
    )

    # ── Text statistics ───────────────────────────────────────────────────
    text_len        = len(text)
    word_count      = len(text.split())
    bullet_points   = text.count("Bullet Point")
    has_description = "product description" in text_lower

    bulk_keywords = ["bulk","wholesale","family size","economy size","jumbo","case","pack of"]
    bulk_score    = sum(1 for k in bulk_keywords if k in text_lower)

    high_price_kw = ["imported","certified","professional","gourmet","premium"]
    low_price_kw  = ["value pack","economy","basic"]
    price_tier    = (
        sum(1 for k in high_price_kw if k in text_lower)
        - sum(1 for k in low_price_kw if k in text_lower)
    )

    qty_keywords = ["set","bundle","combo","kit","pack of","variety"]
    qty_score    = sum(1 for k in qty_keywords if k in text_lower)

    unit_per_oz = weight / (count + 1e-8) if weight > 0 else 0
    unit_per_ml = volume / (count + 1e-8) if volume > 0 else 0

    has_known_brand = any(
        b in text[:150]
        for b in ["Goya","Bush","Starbucks","Lavazza","Celestial","Community Coffee",
                  "Arizona","Planters","Amy","Crystal Light","V8","Snyder"]
    )
    is_single_serve  = any(k in text_lower for k in ["single serve","1 count","individual"])
    is_family_size   = any(k in text_lower for k in ["family size","jumbo","economy"])
    has_special_diet = any(
        k in text_lower
        for k in ["gluten free","vegan","organic","non-gmo","kosher","dairy free"]
    )

    return {
        "weight": weight, "volume": volume, "count": count,
        "extracted_value": extracted_value, "brand_score": brand_score,
        "premium_score": premium_score, "budget_score": budget_score,
        "is_food":        float(dominant_cat == "food"),
        "is_electronics": float(dominant_cat == "electronics"),
        "is_beauty":      float(dominant_cat == "beauty"),
        "is_home":        float(dominant_cat == "home"),
        "is_health":      float(dominant_cat == "health"),
        "is_spices":      float(dominant_cat == "spices"),
        "text_len": text_len, "word_count": word_count,
        "bullet_points": bullet_points, "has_description": float(has_description),
        "bulk_score": bulk_score, "price_tier": price_tier, "qty_score": qty_score,
        "has_weight": float(weight > 0), "has_volume": float(volume > 0),
        "has_count": float(count > 1),
        "unit_per_oz": unit_per_oz, "unit_per_ml": unit_per_ml,
        "has_known_brand":  float(has_known_brand),
        "is_single_serve":  float(is_single_serve),
        "is_family_size":   float(is_family_size),
        "has_special_diet": float(has_special_diet),
        "total_quantity":   weight + volume + (count * 10),
        "weight_to_count":  weight / (count + 1e-8),
        "volume_to_count":  volume / (count + 1e-8),
        "value_to_count":   extracted_value / (count + 1e-8),
        "brand_premium_interaction": brand_score * premium_score,
        "category_quantity": cat_scores.get(dominant_cat, 0) * count,
        "text_richness":  bullet_points + float(has_description) * 2,
        "price_signal":   premium_score * 2 - budget_score,
        "pack_size_score": count * (1 + float(is_family_size)),
        "quality_indicator": premium_score + float(has_special_diet) + float(has_known_brand),
    }


def features_to_tensor(features: dict) -> np.ndarray:
    """Normalise feature dict → 40-dim float32 array (copied from notebook)."""
    return np.array([
        np.log1p(features["weight"] / 10.0),
        np.log1p(features["volume"] / 100.0),
        np.log1p(features["count"]),
        np.log1p(features["extracted_value"] / 10.0),
        features["brand_score"] / 5.0,
        features["premium_score"] / 4.0,
        features["budget_score"] / 3.0,
        features["is_food"],
        features["is_electronics"],
        features["is_beauty"],
        features["is_home"],
        features["is_health"],
        features["is_spices"],
        np.log1p(features["text_len"] / 100.0),
        features["word_count"] / 100.0,
        features["bullet_points"] / 10.0,
        features["has_description"],
        features["bulk_score"] / 3.0,
        features["price_tier"] / 5.0,
        features["qty_score"] / 3.0,
        features["has_weight"],
        features["has_volume"],
        features["has_count"],
        np.log1p(features["unit_per_oz"]),
        np.log1p(features["unit_per_ml"]),
        features["has_known_brand"],
        features["is_single_serve"],
        features["is_family_size"],
        features["has_special_diet"],
        np.log1p(features["total_quantity"] / 100.0),
        np.log1p(features["weight_to_count"]),
        np.log1p(features["volume_to_count"]),
        np.log1p(features["value_to_count"]),
        features["brand_premium_interaction"] / 10.0,
        np.log1p(features["category_quantity"]),
        features["text_richness"] / 10.0,
        np.tanh(features["price_signal"] / 5.0),
        np.log1p(features["pack_size_score"]),
        features["quality_indicator"] / 5.0,
        features["premium_score"] * features["has_known_brand"],
    ], dtype=np.float32)


# =============================================================================
# MODEL CLASSES  (copied exactly from notebook cell_head + cell_model)
# =============================================================================

class EnhancedPriceHead(nn.Module):
    def __init__(self, embed_dim=768, num_features=40):
        super().__init__()
        total_dim = embed_dim * 2 + num_features  # 1576

        self.feature_attention = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.GELU(),
            nn.Linear(num_features, num_features),
            nn.Sigmoid(),
        )

        self.fc1 = nn.Linear(total_dim, 1536); self.ln1 = nn.LayerNorm(1536); self.drop1 = nn.Dropout(0.35)
        self.fc2 = nn.Linear(1536, 768);       self.ln2 = nn.LayerNorm(768);  self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(768, 384);        self.ln3 = nn.LayerNorm(384);  self.drop3 = nn.Dropout(0.3)
        self.res_proj = nn.Linear(total_dim, 384); self.res_ln = nn.LayerNorm(384)

        self.aux1      = nn.Linear(total_dim, 768); self.aux_ln1   = nn.LayerNorm(768);  self.aux_drop1 = nn.Dropout(0.3)
        self.aux2      = nn.Linear(768, 384);       self.aux_ln2   = nn.LayerNorm(384)

        self.fusion_attn = nn.MultiheadAttention(384, num_heads=8, batch_first=True, dropout=0.2)
        self.fusion      = nn.Linear(768, 384); self.fusion_ln = nn.LayerNorm(384); self.fusion_drop = nn.Dropout(0.25)

        self.fc_out   = nn.Linear(384, 192); self.drop_out = nn.Dropout(0.2)
        self.final    = nn.Linear(192, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, clip_feats, tab_feats):
        tab_w = tab_feats * self.feature_attention(tab_feats)
        x     = torch.cat([clip_feats, tab_w], dim=1)

        rp = self.drop1(F.gelu(self.ln1(self.fc1(x))))
        rp = self.drop2(F.gelu(self.ln2(self.fc2(rp))))
        rp = self.drop3(F.gelu(self.ln3(self.fc3(rp))))
        rs = F.gelu(self.res_ln(self.res_proj(x)))
        main = rp + 0.3 * rs

        aux = self.aux_drop1(F.gelu(self.aux_ln1(self.aux1(x))))
        aux = F.gelu(self.aux_ln2(self.aux2(aux)))

        attn_out, _ = self.fusion_attn(main.unsqueeze(1), aux.unsqueeze(1), aux.unsqueeze(1))
        attn_out    = attn_out.squeeze(1)
        fused = self.fusion_drop(F.gelu(self.fusion_ln(self.fusion(torch.cat([attn_out, aux], 1)))))

        return self.final(self.drop_out(F.gelu(self.fc_out(fused))))


class OptimizedCLIPPriceModel(nn.Module):
    def __init__(self, clip_model, embed_dim=768):
        super().__init__()
        self.clip = clip_model

        for name, param in self.clip.visual.named_parameters():
            if "transformer.resblocks." in name:
                try:
                    bn = int(name.split("transformer.resblocks.")[1].split(".")[0])
                    param.requires_grad = bn >= 18
                except:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        for name, param in self.clip.transformer.named_parameters():
            if "resblocks." in name:
                try:
                    bn = int(name.split("resblocks.")[1].split(".")[0])
                    param.requires_grad = bn >= 8
                except:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        self.price_head = EnhancedPriceHead(embed_dim, num_features=40)
        lora_cfg = LoraConfig(
            r=48, lora_alpha=96,
            target_modules=["fc1","fc2","fc3","aux1","aux2","fusion","fc_out","final"],
            lora_dropout=0.15, bias="none",
        )
        self.price_head = get_peft_model(self.price_head, lora_cfg)

    def forward(self, images, text_tokens, tab_feats):
        img_f  = self.clip.encode_image(images)
        text_f = self.clip.encode_text(text_tokens)
        img_f  = img_f  / img_f.norm(dim=-1, keepdim=True)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        return self.price_head(torch.cat([img_f, text_f], dim=1), tab_feats)


# =============================================================================
# MODEL LOADING  (once at import time)
# =============================================================================

print(" Loading CLIP ViT-L/14 …")
_device = torch.device("cpu")
_clip_model, _, _preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
_tokenizer = open_clip.get_tokenizer("ViT-L-14")

print("Loading best_model.pt …")
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pt")
_model = OptimizedCLIPPriceModel(_clip_model, embed_dim=768)

ckpt = torch.load(_MODEL_PATH, map_location="cpu")
# Support both raw state_dict and checkpoint dict saved by training loop
state = ckpt.get("state_dict", ckpt)
_model.load_state_dict(state)
_model.eval()
print(f" Model ready  |  log_mean={LOG_MEAN:.4f}  log_std={LOG_STD:.4f}")


# =============================================================================
# PUBLIC INFERENCE FUNCTION
# =============================================================================

def predict_price(catalog_content: str, image_bytes: bytes) -> dict:
    """
    Run end-to-end inference for one product.

    Returns:
        predicted_price  : float  (USD)
        features         : dict   (40 raw features — passed to LLM for grounding)
        top_signals      : dict   (human-readable top signals for Gemini prompt)
    """
    # 1. Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    img_tensor = _preprocess(img).unsqueeze(0).to(_device)

    # 2. Text tokens
    text_tokens = _tokenizer([catalog_content[:600]]).to(_device)

    # 3. Tabular features
    raw_features   = extract_comprehensive_features(catalog_content)
    tabular_array  = features_to_tensor(raw_features)
    tabular_tensor = torch.tensor(tabular_array).unsqueeze(0).to(_device)

    # 4. Forward pass
    with torch.no_grad():
        log_price_norm = _model(img_tensor, text_tokens, tabular_tensor)

    # 5. Denormalise → price
    log_price       = log_price_norm.item() * LOG_STD + LOG_MEAN
    predicted_price = float(max(np.expm1(log_price), 0.01))

    # 6. Human-readable top signals for Gemini prompt
    top_signals = {
        "pack_count":       raw_features["count"],
        "premium_score":    raw_features["premium_score"],
        "bulk_score":       raw_features["bulk_score"],
        "brand_score":      raw_features["brand_score"],
        "has_special_diet": raw_features["has_special_diet"],
        "quality_indicator":raw_features["quality_indicator"],
        "price_tier":       raw_features["price_tier"],
        "category":         _dominant_category(raw_features),
    }

    return {
        "predicted_price": round(predicted_price, 2),
        "features":        raw_features,
        "top_signals":     top_signals,
    }


def _dominant_category(f: dict) -> str:
    cats = {
        "food": f["is_food"], "electronics": f["is_electronics"],
        "beauty": f["is_beauty"], "home": f["is_home"],
        "health": f["is_health"], "spices": f["is_spices"],
    }
    best = max(cats, key=cats.get)
    return best if cats[best] > 0 else "general"