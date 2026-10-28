# PlacidWay Website Knowledge Chatbot

RAG demo using FastAPI, Sentence Transformers, ChromaDB, and the Anthropic Claude API.

## Prerequisites
- Python 3.11+
- Node.js 20+
- Anthropic API access and an API key
- Exact seven source URLs from the assessment

## Configure backend (Windows PowerShell)
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```
Edit `backend/.env` and set `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, and a long random `ADMIN_API_KEY`.

Add only the exact approved URLs to `backend/app/sources.yaml`. Ingestion honors robots.txt and fails closed when robots access cannot be checked. Review each target host and applicable site terms before ingestion.

Run API:
```powershell
uvicorn app.main:app --reload
```
API docs: http://localhost:8000/docs

In another terminal, configure frontend:
```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```
Frontend: http://localhost:5173

## Ingest approved sources
Use `/api/admin/refresh` in API docs. Include header `X-Admin-Key` with your configured admin key. Poll `/api/admin/refresh/{job_id}`. Refresh is in-process and intended for local evaluation only; use a durable queue and persistent job tracking for multi-worker production.

## Architecture
Approved page fetch -> extraction -> overlapping chunks -> Sentence Transformers vectors -> ChromaDB persistent store -> similarity retrieval -> evidence prompt -> Claude -> answer with source metadata.

## Limitations / hardening before public deployment
- Configure exact URLs; no assessment URLs are assumed.
- Review crawler permissions, legal terms and robots rules.
- The URL fetcher uses public IP checks, but DNS rebinding/TOCTOU and complex proxy/CDN behavior require hardened network egress controls for public operation.
- Tune retrieval thresholds against a labeled evaluation set; cosine scores vary by corpus/model.
- Add durable background queue, admin audit log, session storage/retention policy, monitoring, rate limiting, and deployment secrets management.
- The demo returns retrieved sources alongside answers; frontend displays those links. The model's inline source labels are not parsed into per-claim citations.
- Keep persistent Chroma data on a durable volume.
- Do not present website content as medical advice.
- Run an actual evaluation and record outcomes before submission.

## Tests
```powershell
cd backend
pytest
```
