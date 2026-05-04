"""
api.py
======
TrackMind AI — FastAPI backend.

Exposes retrieval, reasoning and audit as REST endpoints so the HTML
frontend can call them directly. Replaces the Streamlit server.

Run:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

    # Mock mode (no Gemini key needed):
    MOCK=1 uvicorn api:app --reload --port 8000

    # Then open: http://localhost:8000
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="TrackMind AI", version="1.0.0")

# Allow the HTML frontend (served from any origin during dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_MOCK = os.environ.get("MOCK", "").lower() in ("1", "true", "yes") or not os.environ.get("GEMINI_API_KEY", "")

# ── Lazy-load heavy modules once ──────────────────────────────────────────────

_retrieval = None
_reasoning = None
_audit     = None


def get_retrieval():
    global _retrieval
    if _retrieval is None:
        from retrieval import tri_source_retrieve, format_context_for_llm, COLLECTIONS
        _retrieval = (tri_source_retrieve, format_context_for_llm, COLLECTIONS)
    return _retrieval


def get_reasoning():
    global _reasoning
    if _reasoning is None:
        from reasoning import reason, confidence_gate
        _reasoning = (reason, confidence_gate)
    return _reasoning


def get_audit():
    global _audit
    if _audit is None:
        from audit import (
            log_full_interaction, get_recent_entries, get_stats
        )
        _audit = (log_full_interaction, get_recent_entries, get_stats)
    return _audit


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    query: str
    n_chunks: int = 5
    mock: bool = False


class DecisionRequest(BaseModel):
    query_text: str
    verdict: str
    explanation: str
    recommended_action: str
    confidence_tier: str
    confidence_pct: int
    confidence_reason: str
    citations: list[str]
    raw_response: str
    # retrieval context (top articles for audit)
    tsi_top: str = ""
    nntr_top: str = ""
    spec_top: str = ""
    # assessor fields
    assessor_id: str = "NoBo-Assessor-001"
    decision: str = "APPROVED"          # APPROVED / REJECTED / ESCALATED
    notes: str = ""
    edited_explanation: str = ""
    edited_action: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "mock": _MOCK}


@app.get("/collections")
def collections():
    """Return status and chunk counts for all three ChromaDB collections."""
    import chromadb
    try:
        chroma = chromadb.PersistentClient(path="./chroma_db")
        result = {}
        for name, label, lang in [
            ("tsi_loc_pas", "LOC&PAS TSI",  "EN"),
            ("nntr_france", "Arrêté 2012",   "FR"),
            ("spec_doc",    "IberRail Spec", "EN"),
        ]:
            col = chroma.get_or_create_collection(name)
            cnt = col.count()
            result[name] = {"label": label, "lang": lang, "count": cnt, "ok": cnt > 0}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyse")
def analyse(req: AnalyseRequest):
    """
    Full pipeline: retrieve → reason → confidence gate.
    Returns everything the frontend needs to render the result.
    """
    mock = req.mock or _MOCK
    try:
        tri_source_retrieve, format_context_for_llm, _ = get_retrieval()
        reason, confidence_gate = get_reasoning()

        # Retrieve
        retrieval = tri_source_retrieve(req.query, n=req.n_chunks)
        context   = format_context_for_llm(retrieval)

        # Reason
        response = reason(req.query, n_chunks=req.n_chunks, mock=mock)

        # Gate
        allow, gate_msg = confidence_gate(response)

        # Serialise retrieval chunks for the frontend document panel
        def serialise_source(data):
            return {
                "label":  data["label"],
                "lang":   data["lang"],
                "empty":  data["empty"],
                "chunks": [
                    {
                        "text":     c["text"][:370],
                        "article":  c["article"],
                        "distance": c["distance"],
                    }
                    for c in data["chunks"]
                ],
            }

        return {
            "verdict":            response.verdict,
            "explanation":        response.explanation,
            "recommended_action": response.recommended_action,
            "confidence_tier":    response.confidence_tier,
            "confidence_pct":     response.confidence_pct,
            "confidence_reason":  response.confidence_reason,
            "citations":          response.citations,
            "raw_response":       response.raw_response,
            "allow_draft":        allow,
            "gate_message":       gate_msg,
            "retrieval": {
                "tsi":  serialise_source(retrieval["tsi"]),
                "nntr": serialise_source(retrieval["nntr"]),
                "spec": serialise_source(retrieval["spec"]),
            },
            # top articles for audit logging (sent back to client, used in /decide)
            "tsi_top":  retrieval["tsi"]["chunks"][0]["article"]  if retrieval["tsi"]["chunks"]  else "",
            "nntr_top": retrieval["nntr"]["chunks"][0]["article"] if retrieval["nntr"]["chunks"] else "",
            "spec_top": retrieval["spec"]["chunks"][0]["article"] if retrieval["spec"]["chunks"] else "",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decide")
def decide(req: DecisionRequest):
    """Log an assessor decision (approve / reject / escalate)."""
    try:
        log_full_interaction, _, _ = get_audit()

        # Reconstruct a minimal object the audit module accepts
        class _Resp:
            pass

        resp = _Resp()
        resp.query              = req.query_text
        resp.verdict            = req.verdict
        resp.explanation        = req.edited_explanation or req.explanation
        resp.recommended_action = req.edited_action or req.recommended_action
        resp.confidence_tier    = req.confidence_tier
        resp.confidence_pct     = req.confidence_pct
        resp.confidence_reason  = req.confidence_reason
        resp.citations          = req.citations
        resp.raw_response       = req.raw_response

        edited = ""
        if req.edited_explanation or req.edited_action:
            edited = (
                f"EXPLANATION:\n{req.edited_explanation}\n\n"
                f"RECOMMENDED ACTION:\n{req.edited_action}"
            )

        ids = log_full_interaction(
            query_text=req.query_text,
            response=resp,
            assessor_id=req.assessor_id,
            decision=req.decision,
            notes=req.notes,
            edited_response=edited,
            tsi_top=req.tsi_top,
            nntr_top=req.nntr_top,
            spec_top=req.spec_top,
            mock=_MOCK,
        )
        return {"status": "logged", **ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/recent")
def audit_recent(n: int = 5):
    """Return the n most recent audit log entries."""
    try:
        _, get_recent_entries, _ = get_audit()
        return get_recent_entries(n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/stats")
def audit_stats():
    """Return summary statistics for the sidebar."""
    try:
        _, _, get_stats = get_audit()
        return get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve the HTML frontend ───────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/{filepath:path}")
def static_file(filepath: str):
    """Serve static files (logo, etc.) from the same directory."""
    if os.path.isfile(filepath) and not filepath.startswith("."):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Not found")
