
import asyncio
import secrets
import uuid
import logging
import re

from app.services.knowledge_loader import load_knowledge_export
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from anthropic import APIError

from app.config import settings
from app.schemas import ChatRequest, ChatResponse, Citation
from app.services.rag_service import retrieve, active_collection
from app.services.llm_service import generate_answer
from app.services.ingestion_service import read_sources, ingest_all



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PlacidWay Knowledge Chatbot",
    version="0.1.0"
)
@app.on_event("startup")
def load_bundled_knowledge():
    try:
        count = load_knowledge_export()
        logger.info("Startup knowledge check completed. Chunks available: %s", count)
    except Exception:
        logger.exception("Failed to load bundled knowledge")


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

# Read origins configured in backend settings.
configured_origins = getattr(
    settings,
    "allowed_origins",
    ""
) or ""

allowed_origins = {
    origin.strip().rstrip("/")
    for origin in configured_origins.split(",")
    if origin.strip()
}

# Allow the local React/Vite development server.
local_origins = {
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
}

allowed_origins.update(local_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "X-Admin-Key",
        "Authorization",
    ],
)


# --------------------------------------------------
# ADMIN JOB STORAGE
# --------------------------------------------------

jobs: dict[str, dict] = {}


def require_admin(key: str | None):
    admin_key = getattr(
        settings,
        "admin_api_key",
        None
    )

    if (
        not admin_key
        or not key
        or not secrets.compare_digest(
            key,
            admin_key
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


# --------------------------------------------------
# APPOINTMENT REQUEST DETECTION
# --------------------------------------------------

def is_appointment_request(question: str) -> bool:
    """Identify requests to book, schedule, or confirm appointments."""

    normalized = re.sub(
        r"\s+",
        " ",
        question.lower()
    ).strip()

    appointment_patterns = [
        r"\bbook\b.{0,50}\bappointment\b",
        r"\bschedule\b.{0,50}\bappointment\b",
        r"\bmake\b.{0,30}\bappointment\b",
        r"\bappointment\b.{0,50}\b(book|schedule|tomorrow)\b",
        r"\bcan you\b.{0,30}\b(book|schedule)\b",
        r"\bcheck\b.{0,30}\bappointment availability\b",
        r"\bconfirm\b.{0,30}\bappointment\b",
        r"\bappointment\b.{0,30}\bfor tomorrow\b",
        r"\bcan i get an appointment\b",
    ]

    return any(
        re.search(pattern, normalized)
        for pattern in appointment_patterns
    )


# --------------------------------------------------
# MEDICAL SAFETY QUESTION DETECTION
# --------------------------------------------------

def is_medical_safety_question(question: str) -> bool:
    """Identify personal medical safety and risk questions."""

    q = re.sub(
        r"\s+",
        " ",
        question.lower()
    ).strip()

    safety_patterns = [
        r"\bis .{0,40} safe for me\b",
        r"\bis .{0,40} safe for my\b",
        r"\bsafe with my\b",
        r"\bsafe if i have\b",
        r"\bcan i .{0,50} with my\b",
        r"\bcan i .{0,50} if i have\b",
        r"\brisks? .{0,40} for my\b",
        r"\bmy .{0,30} (condition|disease)\b",
        r"\bwill .{0,30} affect my\b",
        r"\bshould i .{0,30} with my\b",
    ]

    medical_terms = [
        "diabetes",
        "surgery",
        "surgical",
        "operation",
        "treatment",
        "procedure",
        "heart disease",
        "blood pressure",
        "kidney disease",
        "cancer",
        "medical condition",
        "medication",
    ]

    return (
        any(
            re.search(pattern, q)
            for pattern in safety_patterns
        )
        and any(
            term in q
            for term in medical_terms
        )
    )


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/api/health")
async def health():
    try:
        count = active_collection().count()
    except Exception:
        logger.exception(
            "Unable to read indexed chunk count"
        )
        count = 0

    return {
        "status": "ok",
        "indexed_chunks": count
    }


# --------------------------------------------------
# SOURCES ENDPOINT
# --------------------------------------------------

@app.get("/api/sources")
async def sources():
    try:
        items = read_sources()

        return [
            {
                "title": source["title"],
                "configured": bool(
                    source.get("url", "").strip()
                )
            }
            for source in items
        ]

    except Exception:
        logger.exception(
            "Unable to read source registry"
        )

        raise HTTPException(
            status_code=500,
            detail="Source registry is unavailable."
        )


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.post(
    "/api/chat",
    response_model=ChatResponse
)
async def chat(req: ChatRequest):

    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question is required."
        )

    # Handle appointment requests before retrieval
    # or Claude generation.
    if is_appointment_request(question):
        return ChatResponse(
            answer=(
                "I can't book appointments or check availability. "
                "To request an appointment, please contact the clinic "
                "directly or use the quote request option on the "
                "relevant PlacidWay page, if available. "
                "Your appointment is not confirmed until the clinic "
                "confirms the date and time."
            ),
            sources=[],
            grounded=False
        )

    # Handle personal medical-safety questions.
    if is_medical_safety_question(question):
        return ChatResponse(
            answer=(
                "I can't determine whether a surgery or treatment is "
                "safe for you based on your personal medical condition. "
                "Please consult your treating doctor and the surgeon "
                "before making a decision. The reviewed PlacidWay pages "
                "do not contain enough information to assess your "
                "individual medical risks."
            ),
            sources=[],
            grounded=False
        )

    # Retrieve relevant chunks from the knowledge base.
    try:
        hits = await asyncio.to_thread(
            retrieve,
            question
        )

    except Exception:
        logger.exception(
            "Retrieval failed"
        )

        raise HTTPException(
            status_code=503,
            detail="Knowledge search is temporarily unavailable."
        )

    # Do not generate an answer if no evidence is found.
    if not hits:
        return ChatResponse(
            answer=(
                "I couldn't find enough information in the approved "
                "PlacidWay pages to answer that reliably. You can "
                "contact the clinic or request a quote through a "
                "verified option on the relevant PlacidWay page."
            ),
            sources=[],
            grounded=False
        )

    evidence_parts = []
    citations = []
    source_labels = {}

    # Assign one source label per unique URL.
    for hit in hits:
        metadata = hit.get("metadata") or {}

        url = metadata.get("url")
        title = metadata.get("title") or "Approved source"

        # Skip chunks without a source URL.
        if not url:
            continue

        # Reuse label for chunks from the same page.
        if url not in source_labels:
            label = f"S{len(source_labels) + 1}"
            source_labels[url] = label

            citations.append(
                Citation(
                    title=title,
                    url=url
                )
            )
        else:
            label = source_labels[url]

        evidence_parts.append(
            f"[{label}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {hit.get('text', '')}"
        )

    # Fail safely if no usable evidence remains.
    if not evidence_parts:
        return ChatResponse(
            answer=(
                "I couldn't find enough reliably sourced information "
                "in the approved PlacidWay pages. Please contact "
                "the clinic or request a quote for confirmation."
            ),
            sources=[],
            grounded=False
        )

    evidence = "\n\n".join(evidence_parts)

    # Generate the answer from retrieved evidence.
    try:
        answer = await generate_answer(
            question,
            evidence,
            req.history
        )

    except (APIError, TimeoutError):
        logger.exception(
            "Anthropic request failed"
        )

        raise HTTPException(
            status_code=502,
            detail="The answer service is temporarily unavailable."
        )

    except RuntimeError:
        logger.exception(
            "Answer generation failed"
        )

        raise HTTPException(
            status_code=503,
            detail="Answer generation failed. Please try again."
        )

    # Keep only source links referenced by valid source labels.
    referenced_labels = {
        int(number)
        for number in re.findall(
            r"\[S(\d+)\]",
            answer
        )
    }

    citations = [
        citation
        for index, citation in enumerate(
            citations,
            start=1
        )
        if index in referenced_labels
    ]

    return ChatResponse(
        answer=answer,
        sources=citations,
        grounded=True
    )


# --------------------------------------------------
# KNOWLEDGE BASE REFRESH
# --------------------------------------------------

async def run_refresh(job_id: str):
    jobs[job_id] = {
        "status": "running"
    }

    try:
        result = await ingest_all()

        jobs[job_id] = {
            "status": "completed",
            "result": result
        }

    except Exception as e:
        logger.exception(
            "Refresh failed"
        )

        jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }


# --------------------------------------------------
# ADMIN REFRESH ENDPOINT
# --------------------------------------------------

@app.post(
    "/api/admin/refresh",
    status_code=202
)
async def refresh(
    background: BackgroundTasks,
    x_admin_key: str | None = Header(default=None)
):
    require_admin(x_admin_key)

    job_id = uuid.uuid4().hex

    jobs[job_id] = {
        "status": "queued"
    }

    background.add_task(
        run_refresh,
        job_id
    )

    return {
        "job_id": job_id,
        "status": "queued"
    }


# --------------------------------------------------
# ADMIN REFRESH STATUS
# --------------------------------------------------

@app.get("/api/admin/refresh/{job_id}")
async def refresh_status(
    job_id: str,
    x_admin_key: str | None = Header(default=None)
):
    require_admin(x_admin_key)

    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    return jobs[job_id]
