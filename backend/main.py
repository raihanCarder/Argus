import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

load_dotenv(find_dotenv())

DATA_DIR = Path(__file__).parent / "data"
SIGNALS_PATH = DATA_DIR / "signals.json"

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_FALLBACK_MODEL = os.getenv("EMBEDDING_FALLBACK_MODEL", "")
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
MIN_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "0.5"))
KEYWORD_ONLY_MIN_MATCH_SCORE = float(os.getenv("KEYWORD_ONLY_MIN_MATCH_SCORE", "0.2"))
REASONING_TEMPERATURE = float(os.getenv("REASONING_TEMPERATURE", "0.3"))
MAX_REASONING_TOKENS = int(os.getenv("MAX_REASONING_TOKENS", "100"))
MAX_MATCHES = int(os.getenv("MAX_MATCHES", "10"))
NO_MATCH_FALLBACK_TOP_K = int(os.getenv("NO_MATCH_FALLBACK_TOP_K", "5"))
EMAIL_TEMPERATURE = float(os.getenv("EMAIL_TEMPERATURE", "0.7"))
MAX_EMAIL_TOKENS = int(os.getenv("MAX_EMAIL_TOKENS", "400"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
PRECOMPUTE_SIGNAL_EMBEDDINGS = os.getenv("PRECOMPUTE_SIGNAL_EMBEDDINGS", "false").lower() == "true"
SCRAPE_ALWAYS_LIVE = os.getenv("SCRAPE_ALWAYS_LIVE", "true").lower() == "true"
SCRAPE_CACHE_SECONDS = int(os.getenv("SCRAPE_CACHE_SECONDS", "3600"))
ALLOW_FILE_FALLBACK = os.getenv("ALLOW_FILE_FALLBACK", "false").lower() == "true"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="DealFlow-AI")
origins = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
allow_credentials = True
if len(origins) == 1 and origins[0] == "*":
    allow_credentials = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    startup_description: str


class EmailRequest(BaseModel):
    startup_description: str
    signal: dict
    match_score: float


@lru_cache(maxsize=1)
def load_signals() -> tuple[dict[str, Any], ...]:
    if not SIGNALS_PATH.exists():
        return tuple()
    with SIGNALS_PATH.open("r", encoding="utf-8") as f:
        return tuple(json.load(f))


EMBED_CACHE: dict[str, list[float]] = {}
LIVE_SIGNALS_CACHE: list[dict[str, Any]] = []
LIVE_SIGNALS_TS = 0.0


def get_signal_text(signal: dict[str, Any]) -> str:
    return f"{signal.get('category', '')} {signal.get('title', '')} {signal.get('description', '')}"


def get_live_signals(startup_hint: str = "") -> tuple[list[dict[str, Any]], str, str | None]:
    global LIVE_SIGNALS_CACHE, LIVE_SIGNALS_TS
    now = time.time()
    cache_fresh = LIVE_SIGNALS_CACHE and (now - LIVE_SIGNALS_TS) < SCRAPE_CACHE_SECONDS

    if not SCRAPE_ALWAYS_LIVE and cache_fresh:
        return LIVE_SIGNALS_CACHE, "cache", None

    scrape_error: str | None = None
    try:
        # Import lazily so scrape_signals sees env vars after load_dotenv().
        from scrape_signals import collect_signals, get_last_collect_meta

        signals = collect_signals(startup_hint=startup_hint)
        collect_meta = get_last_collect_meta()
        if signals:
            LIVE_SIGNALS_CACHE = signals
            LIVE_SIGNALS_TS = now
            if collect_meta.get("error"):
                return signals, "live", str(collect_meta.get("error"))
            return signals, "live", None
        scrape_error = str(collect_meta.get("error") or "Live scrape returned 0 signals.")
    except Exception as exc:
        scrape_error = f"Live scrape failed: {type(exc).__name__}: {exc}"

    if cache_fresh:
        return LIVE_SIGNALS_CACHE, "cache", scrape_error

    if ALLOW_FILE_FALLBACK:
        file_signals = list(load_signals())
        if file_signals:
            return file_signals, "file", scrape_error

    return [], "none", scrape_error


def get_embedding(text: str) -> list[float]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY missing")
    cached = EMBED_CACHE.get(text)
    if cached is not None:
        return cached
    try:
        model_name = EMBEDDING_MODEL
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_document",
        )
        embedding = result["embedding"]
        EMBED_CACHE[text] = embedding
        return embedding
    except Exception as exc:  # pragma: no cover - external API error surface
        if EMBEDDING_FALLBACK_MODEL:
            try:
                fallback_model = EMBEDDING_FALLBACK_MODEL
                if not fallback_model.startswith("models/"):
                    fallback_model = f"models/{fallback_model}"
                result = genai.embed_content(
                    model=fallback_model,
                    content=text,
                    task_type="retrieval_document",
                )
                embedding = result["embedding"]
                EMBED_CACHE[text] = embedding
                return embedding
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc


def calculate_keyword_overlap(startup_desc: str, signal_keywords: list[str]) -> float:
    def normalize_tokens(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\\-]+", text.lower())

    startup_words = set(normalize_tokens(startup_desc))
    keyword_words = set(normalize_tokens(" ".join(signal_keywords)))

    if not keyword_words:
        return 0.0

    match_count = 0
    for keyword in keyword_words:
        if keyword in startup_words:
            match_count += 1
            continue
        # Allow partial token matches for related terms (e.g., cyber vs cybersecurity).
        if any(
            (keyword.startswith(sw[:4]) or sw.startswith(keyword[:4]))
            for sw in startup_words
            if len(sw) >= 4 and len(keyword) >= 4
        ):
            match_count += 1

    return match_count / len(keyword_words)


def generate_match_reasoning(startup_desc: str, signal: dict, score: float) -> str:
    if not GEMINI_API_KEY:
        return ""

    budget_value = signal.get("budget")
    if isinstance(budget_value, (int, float)):
        budget_text = f"${budget_value:,.0f}"
    else:
        budget_text = "Unknown"

    prompt = f"""You are an expert at matching GovTech startups to government procurement opportunities.

STARTUP:
{startup_desc}

GOVERNMENT SIGNAL:
- Category: {signal.get('category', '')}
- Title: {signal.get('title', '')}
- Description: {signal.get('description', '')}
- Budget: {budget_text}
- Key Requirements: {', '.join(signal.get('keywords', []))}

Match Score: {score:.0%}

Task: Write exactly 2 sentences explaining why this startup matches this opportunity.

Requirements:
- Sentence 1: State the specific startup capability that addresses the government need
- Sentence 2: Mention the budget, timeline, or a stakeholder to show you understand the context
- Be concrete and specific
- No fluff or generic statements
- Maximum 50 words total
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=REASONING_TEMPERATURE,
                max_output_tokens=MAX_REASONING_TOKENS,
            ),
        )
        text = (response.text or "").strip()
        if len(text.split()) < 8:
            return (
                f"This signal aligns at {score:.0%} based on overlap with your startup capabilities. "
                f"It references {signal.get('city', 'a city')} and the initiative '{signal.get('title', 'opportunity')}'."
            )
        return text
    except Exception:
        return (
            f"This signal aligns at {score:.0%} based on overlap with your startup capabilities. "
            f"It references {signal.get('city', 'a city')} and the initiative '{signal.get('title', 'opportunity')}'."
        )


def generate_weak_fit_reasoning(signal: dict, score: float) -> str:
    title = signal.get("title", "this signal")
    city = signal.get("city", "the city")
    return (
        f"This is a weaker-fit live signal ({score:.0%}) from {city} for {title}. "
        "It is included as a fallback so you can inspect potential stakeholders and timing."
    )


def generate_fallback_reasoning(signal: dict) -> str:
    city = signal.get("city", "the city")
    source_title = signal.get("source_title", "source")
    return (
        f"This is a placeholder signal generated from {source_title} in {city}. "
        "Treat it as directional only until a source-grounded signal is extracted."
    )


@app.on_event("startup")
def warm_cache() -> None:
    if not USE_EMBEDDINGS:
        return
    signals = load_signals()
    if not PRECOMPUTE_SIGNAL_EMBEDDINGS:
        return
    for signal in signals:
        signal_text = get_signal_text(signal)
        if signal_text not in EMBED_CACHE and "embedding" in signal:
            EMBED_CACHE[signal_text] = signal["embedding"]
        elif signal_text not in EMBED_CACHE:
            _ = get_embedding(signal_text)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "apis": {
            "google": bool(GEMINI_API_KEY),
            "serper": bool(os.getenv("SERPER_KEY", "") or os.getenv("SERPER_API_KEY", "")),
            "jina": bool(os.getenv("JINA_KEY", "") or os.getenv("JINA_API_KEY", "")),
        },
        "embedding_model": EMBEDDING_MODEL,
        "use_embeddings": USE_EMBEDDINGS,
    }


@app.post("/match")
def match_startup(request: MatchRequest) -> dict[str, Any]:
    signals, data_source, scrape_error = get_live_signals(startup_hint=request.startup_description)
    if not signals:
        response: dict[str, Any] = {"matches": [], "data_source": data_source}
        if scrape_error:
            response["scrape_error"] = scrape_error
        return response

    startup_embedding: list[float] | None = None
    embedding_error: str | None = None
    if GEMINI_API_KEY and USE_EMBEDDINGS:
        try:
            startup_embedding = get_embedding(request.startup_description)
        except HTTPException as exc:
            embedding_error = str(exc.detail)
            startup_embedding = None
    scored_candidates: list[dict[str, Any]] = []

    for idx, signal in enumerate(signals):
        signal_text = get_signal_text(signal)
        semantic_score = 0.0
        if startup_embedding is not None:
            signal_embedding = signal.get("embedding")
            if not signal_embedding:
                try:
                    signal_embedding = get_embedding(signal_text)
                except HTTPException as exc:
                    embedding_error = str(exc.detail)
                    signal_embedding = None
            if signal_embedding is not None:
                import math
                raw_sim = cosine_similarity(
                    [startup_embedding],
                    [signal_embedding],
                )[0][0]
                semantic_score = 0.0 if (math.isnan(raw_sim) or math.isinf(raw_sim)) else float(raw_sim)

        keyword_corpus = list(signal.get("keywords", []))
        keyword_corpus.extend(
            [
                str(signal.get("title", "")),
                str(signal.get("description", "")),
                str(signal.get("category", "")),
            ]
        )
        keyword_score = calculate_keyword_overlap(
            request.startup_description.lower(),
            keyword_corpus,
        )

        weight_sum = SEMANTIC_WEIGHT + KEYWORD_WEIGHT
        if weight_sum <= 0:
            final_score = 0.0
        else:
            if startup_embedding is None:
                final_score = keyword_score
            else:
                final_score = (semantic_score * SEMANTIC_WEIGHT + keyword_score * KEYWORD_WEIGHT) / weight_sum

        if signal.get("_source_verified") is False:
            final_score = max(0.0, final_score - 0.05)

        signal_payload = {k: v for k, v in signal.items() if k != "embedding"}
        # Ensure frontend-compatible fields
        if "id" not in signal_payload:
            signal_payload["id"] = idx
        if "state" not in signal_payload:
            signal_payload["state"] = signal_payload.get("region", "")
        # Ensure lat/lng are numbers (not None) for the 3D globe
        if not isinstance(signal_payload.get("lat"), (int, float)):
            signal_payload["lat"] = 0
        if not isinstance(signal_payload.get("lng"), (int, float)):
            signal_payload["lng"] = 0
        scored_candidates.append(
            {
                "signal": signal_payload,
                "score": float(final_score),
            }
        )

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    threshold = MIN_MATCH_SCORE if startup_embedding is not None else KEYWORD_ONLY_MIN_MATCH_SCORE
    selected = [item for item in scored_candidates if item["score"] >= threshold]
    weak_fit_fallback = False
    if not selected and scored_candidates:
        weak_fit_fallback = True
        selected = scored_candidates[: max(1, NO_MATCH_FALLBACK_TOP_K)]

    matches: list[dict[str, Any]] = []
    for item in selected[:MAX_MATCHES]:
        signal_payload = item["signal"]
        score = item["score"]
        is_fallback_signal = bool(signal_payload.get("_fallback"))
        reasoning = (
            generate_weak_fit_reasoning(signal_payload, score)
            if weak_fit_fallback
            else generate_fallback_reasoning(signal_payload)
            if is_fallback_signal
            else generate_match_reasoning(
                request.startup_description,
                signal_payload,
                score,
            )
        )
        matches.append(
            {
                "signal": signal_payload,
                "score": score,
                "reasoning": reasoning,
            }
        )

    response: dict[str, Any] = {"matches": matches, "data_source": data_source}
    if scrape_error:
        response["scrape_error"] = scrape_error
    if startup_embedding is None and embedding_error:
        response["warning"] = "Embedding fallback to keyword-only matching"
        response["embedding_error"] = embedding_error
    if weak_fit_fallback:
        response["warning"] = "No strong matches found; returning top live weak-fit candidates."
    if any(bool(match["signal"].get("_fallback")) for match in matches):
        response["warning"] = "Some matches are placeholder signals; source-grounded extraction is incomplete."
    return response


@app.post("/email")
def generate_email(request: EmailRequest) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY missing")

    signal = request.signal
    prompt = f"""Write a professional outreach email from a startup to a government contact.

FROM (Startup):
{request.startup_description}

TO (Government Contact):
{signal.get('stakeholders', ['City Official'])[0] if signal.get('stakeholders') else 'City Official'}

REGARDING (Opportunity):
- Title: {signal.get('title', '')}
- Description: {signal.get('description', '')}
- Budget: ${signal.get('budget', 0):,}
- Timeline: {signal.get('timeline', '')}
- Match Score: {request.match_score}%

TASK: Write a concise 3-paragraph email:

Paragraph 1 (Introduction):
- Reference the specific government initiative
- Explain why you're reaching out
- Mention how you learned about this (council minutes, strategic plan, etc.)

Paragraph 2 (Value Proposition):
- Highlight 2-3 specific capabilities that address their needs
- Use concrete metrics if available
- Show you understand their requirements

Paragraph 3 (Call to Action):
- Request a 30-minute introductory call
- Suggest next steps
- Provide availability

REQUIREMENTS:
- Professional but not stiff
- Specific, not generic
- Under 200 words
- No marketing fluff
- Include actual contact info (their email address)
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=EMAIL_TEMPERATURE,
                max_output_tokens=MAX_EMAIL_TOKENS,
            ),
        )
        email_body = response.text.strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Email generation failed: {exc}") from exc

    return {
        "subject": f"Re: {signal.get('title', 'Opportunity')} - Partnership Opportunity",
        "to": signal.get("stakeholders", ["City Official"])[0],
        "body": email_body,
        "preview_note": "This is a preview. Review and customize before sending.",
    }
