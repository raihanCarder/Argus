# Argus

Argus is a demo-first product that helps GovTech startups discover government opportunities before formal RFPs are posted. A user describes their startup, the backend matches it against pre‑seeded government signals, and the UI returns top matches with AI reasoning and a draft outreach email.

## Why It Matters

- Government procurement is slow and opaque. Argus surfaces early signals so startups can engage sooner.
- The MVP prioritizes a WOW demo with strong UX and believable data instead of complex scraping.
- Semantic matching + clear explanations make the results feel credible and actionable.

## Tech Stack

- Frontend: Next.js (App Router), React, Tailwind CSS, Three.js, Framer Motion
- Backend: FastAPI, Uvicorn, Gemini API (embeddings + generation), scikit-learn

## Quick Start

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp ../.env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## Scrape Signals

Signals are collected in real-time when the backend receives a `/match` request.

This pipeline now uses Gemini directly for signal intelligence and link discovery from the startup description. Ensure `GOOGLE_API_KEY` is set in `.env`.

## Environment Setup

This project uses a single root `.env` file. Copy `.env.example` to `.env` at the repo root, then add your Gemini API key. Ask a contributor for a key if you don’t have one.

## Contributors

- Dunura epasingag
- Raihan Carder
- Fatima Reehn
- Akshay Krishna Sirigana
