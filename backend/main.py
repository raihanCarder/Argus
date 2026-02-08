import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
SIGNALS_PATH = DATA_DIR / "signals.json"

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
MIN_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "0.5"))
REASONING_TEMPERATURE = float(os.getenv("REASONING_TEMPERATURE", "0.3"))
MAX_REASONING_TOKENS = int(os.getenv("MAX_REASONING_TOKENS", "100"))
MAX_MATCHES = int(os.getenv("MAX_MATCHES", "10"))
EMAIL_TEMPERATURE = float(os.getenv("EMAIL_TEMPERATURE", "0.7"))
MAX_EMAIL_TOKENS = int(os.getenv("MAX_EMAIL_TOKENS", "400"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
PRECOMPUTE_SIGNAL_EMBEDDINGS = os.getenv("PRECOMPUTE_SIGNAL_EMBEDDINGS", "false").lower() == "true"

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


def get_signal_text(signal: dict[str, Any]) -> str:
    return f"{signal.get('category', '')} {signal.get('title', '')} {signal.get('description', '')}"


def get_embedding(text: str) -> list[float]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY missing")
    cached = EMBED_CACHE.get(text)
    if cached is not None:
        return cached
    try:
        result = genai.embed_content(
            model=f"models/{EMBEDDING_MODEL}",
            content=text,
            task_type="retrieval_document",
        )
        embedding = result["embedding"]
        EMBED_CACHE[text] = embedding
        return embedding
    except Exception as exc:  # pragma: no cover - external API error surface
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc


def calculate_keyword_overlap(startup_desc: str, signal_keywords: list[str]) -> float:
    startup_words = set(startup_desc.lower().split())
    keyword_words = set(" ".join(signal_keywords).lower().split())

    if not keyword_words:
        return 0.0

    matches = startup_words & keyword_words
    return len(matches) / len(keyword_words)


def generate_match_reasoning(startup_desc: str, signal: dict, score: float) -> str:
    if not GEMINI_API_KEY:
        return ""

    prompt = f"""You are an expert at matching GovTech startups to government procurement opportunities.

STARTUP:
{startup_desc}

GOVERNMENT SIGNAL:
- Category: {signal.get('category', '')}
- Title: {signal.get('title', '')}
- Description: {signal.get('description', '')}
- Budget: ${signal.get('budget', 0):,}
- Key Requirements: {', '.join(signal.get('keywords', []))}

Match Score: {score:.0%}

Task: Write exactly 2 sentences explaining why this startup matches this opportunity.

Requirements:
- Sentence 1: State the specific startup capability that addresses the government need
- Sentence 2: Mention the budget/timeline/stakeholder to show you understand the context
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
        return response.text.strip()
    except Exception:
        return ""


@app.on_event("startup")
def warm_cache() -> None:
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
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/match")
def match_startup(request: MatchRequest) -> dict[str, Any]:
    signals = load_signals()
    if not signals:
        return {"matches": []}

    startup_embedding = get_embedding(request.startup_description)
    matches: list[dict[str, Any]] = []

    for signal in signals:
        signal_text = get_signal_text(signal)
        signal_embedding = signal.get("embedding")
        if not signal_embedding:
            signal_embedding = get_embedding(signal_text)

        semantic_score = cosine_similarity(
            [startup_embedding],
            [signal_embedding],
        )[0][0]

        keyword_score = calculate_keyword_overlap(
            request.startup_description.lower(),
            signal.get("keywords", []),
        )

        weight_sum = SEMANTIC_WEIGHT + KEYWORD_WEIGHT
        if weight_sum <= 0:
            final_score = 0.0
        else:
            final_score = (semantic_score * SEMANTIC_WEIGHT + keyword_score * KEYWORD_WEIGHT) / weight_sum

        if final_score >= MIN_MATCH_SCORE:
            reasoning = generate_match_reasoning(
                request.startup_description,
                signal,
                final_score,
            )
            signal_payload = {k: v for k, v in signal.items() if k != "embedding"}
            matches.append(
                {
                    "signal": signal_payload,
                    "score": float(final_score),
                    "reasoning": reasoning,
                }
            )

    matches.sort(key=lambda x: x["score"], reverse=True)
    return {"matches": matches[:MAX_MATCHES]}


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

TASK: Write a concise 3-paragraph email.

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
