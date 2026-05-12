"""
api.py
======
TrackMind — FastAPI backend.

Updated architecture:
  - /upload-spec   NEW: accepts a spec PDF, chunks it in memory, stores in session cache
  - /analyse       UPDATED: uses session spec chunks if available (no ChromaDB write)
  - /decide        unchanged
  - /audit/*       unchanged

The spec PDF is never written to ChromaDB. It lives only in the server's
in-memory session store (_SESSION_SPECS dict, keyed by session_id).
This keeps manufacturer specs off the persistent vector DB.

Run:
    export ANTHROPIC_API_KEY=<your-api-key>
    uvicorn api:app --reload --port 8000

Open: http://localhost:8000
"""

import os
import uuid
from types import SimpleNamespace
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="TrackMind", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session spec store ──────────────────────────────────────────────
# key:   session_id (str UUID, returned at upload time)
# value: list of chunk dicts (from trackmind_chunker.chunk_generic_from_bytes)
_SESSION_SPECS: dict[str, dict] = {}
# dict value shape: { "chunks": [...], "filename": str, "chunk_count": int }


# ── Lazy-load heavy modules once ──────────────────────────────────────────────

_reasoning = None
_audit     = None


def get_reasoning():
    global _reasoning
    if _reasoning is None:
        from reasoning import reason, confidence_gate
        _reasoning = (reason, confidence_gate)
    return _reasoning


def get_audit():
    global _audit
    if _audit is None:
        from audit import log_full_interaction, get_recent_entries, get_stats
        _audit = (log_full_interaction, get_recent_entries, get_stats)
    return _audit


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    query: str
    n_chunks: int = 5
    mock: bool = False
    session_id: str = ""   # links to uploaded spec; empty = regulatory-only mode


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
    tsi_top: str = ""
    nntr_top: str = ""
    spec_top: str = ""
    assessor_id: str = "NoBo-Assessor-001"
    decision: str = "APPROVED"
    notes: str = ""
    edited_explanation: str = ""
    edited_action: str = ""
    mock: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_SESSION_SPECS)}


@app.get("/collections")
def collections():
    """Return status and chunk counts for permanent ChromaDB collections (TSI + NNTR only)."""
    import chromadb
    try:
        chroma = chromadb.PersistentClient(path="./chroma_db")
        result = {}
        for name, label, lang in [
            ("tsi_loc_pas", "LOC&PAS TSI",  "EN"),
            ("nntr_france", "Arrêté 2012",   "FR"),
        ]:
            col = chroma.get_or_create_collection(name)
            cnt = col.count()
            result[name] = {"label": label, "lang": lang, "count": cnt, "ok": cnt > 0}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-spec")
async def upload_spec(file: UploadFile = File(...)):
    """
    Accept a manufacturer spec PDF, chunk it in memory, store in session cache.

    Returns a session_id the frontend passes back in subsequent /analyse calls.
    The spec is NEVER written to ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        from trackmind_chunker import chunk_generic_from_bytes

        pdf_bytes = await file.read()
        chunks = chunk_generic_from_bytes(
            pdf_bytes=pdf_bytes,
            doc_type="SPEC",
            source_name=file.filename,
            language="en",
        )

        from retrieval import embed
        for chunk in chunks:
            chunk["_embedding"] = embed(chunk["text"])

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the uploaded PDF."
            )

        session_id = str(uuid.uuid4())
        _SESSION_SPECS[session_id] = {
            "chunks":      chunks,
            "filename":    file.filename,
            "chunk_count": len(chunks),
        }

        return {
            "session_id":  session_id,
            "filename":    file.filename,
            "chunk_count": len(chunks),
            "message":     (
                f"Spec '{file.filename}' loaded into session memory. "
                f"{len(chunks)} chunks ready. Spec is NOT stored in the vector database."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.delete("/upload-spec/{session_id}")
def clear_spec(session_id: str):
    """Remove a spec from the session cache (e.g. when user uploads a new one)."""
    if session_id in _SESSION_SPECS:
        info = _SESSION_SPECS.pop(session_id)
        return {"cleared": True, "filename": info["filename"]}
    raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/upload-spec/{session_id}")
def spec_info(session_id: str):
    """Return info about a loaded spec session."""
    if session_id not in _SESSION_SPECS:
        raise HTTPException(status_code=404, detail="Session not found.")
    info = _SESSION_SPECS[session_id]
    return {
        "session_id":  session_id,
        "filename":    info["filename"],
        "chunk_count": info["chunk_count"],
    }


@app.post("/analyse")
def analyse(req: AnalyseRequest):
    """
    Full pipeline: retrieve → reason → confidence gate.

    If session_id is provided and valid, uses the session spec chunks for SPEC source.
    Otherwise runs regulatory-only (TSI + NNTR) and notes spec is missing.
    """
    try:
        reason, confidence_gate = get_reasoning()

        # Resolve session spec
        session_spec_chunks = None
        spec_filename = None
        if req.session_id and req.session_id in _SESSION_SPECS:
            session_data = _SESSION_SPECS[req.session_id]
            session_spec_chunks = session_data["chunks"]
            spec_filename = session_data["filename"]

        # reason() now returns (ComplianceResponse, retrieval_dict)
        response, retrieval = reason(
            req.query,
            n_chunks=req.n_chunks,
            mock=req.mock,
            session_spec_chunks=session_spec_chunks,
        )

        # Confidence gate
        allow, gate_msg = confidence_gate(response)

        # Serialise retrieval chunks for the frontend document panel
        def serialise_source(data):
            return {
                "label":  data["label"],
                "lang":   data["lang"],
                "empty":  data["empty"],
                "chunks": [
                    {
                        "text":     c["text"],
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
            "spec_filename":      spec_filename,
            "spec_loaded":        session_spec_chunks is not None,
            "retrieval": {
                "tsi":  serialise_source(retrieval["tsi"]),
                "nntr": serialise_source(retrieval["nntr"]),
                "spec": serialise_source(retrieval["spec"]),
            },
            # top articles for audit logging
            "tsi_top":  retrieval["tsi"]["chunks"][0]["article"]  if retrieval["tsi"]["chunks"]  else "",
            "nntr_top": retrieval["nntr"]["chunks"][0]["article"] if retrieval["nntr"]["chunks"] else "",
            "spec_top": retrieval["spec"]["chunks"][0]["article"] if retrieval["spec"]["chunks"] else "",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decide")
def decide(req: DecisionRequest):
    """Log an assessor decision (approve / reject)."""
    try:
        log_full_interaction, _, _ = get_audit()

        decision = "REJECTED" if req.decision == "ESCALATED" else req.decision

        resp = SimpleNamespace(
            query=req.query_text,
            verdict=req.verdict,
            explanation=req.edited_explanation or req.explanation,
            recommended_action=req.edited_action or req.recommended_action,
            confidence_tier=req.confidence_tier,
            confidence_pct=req.confidence_pct,
            confidence_reason=req.confidence_reason,
            citations=req.citations,
            raw_response=req.raw_response,
        )

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
            decision=decision,
            notes=req.notes,
            edited_response=edited,
            tsi_top=req.tsi_top,
            nntr_top=req.nntr_top,
            spec_top=req.spec_top,
            mock=req.mock,
        )
        return {"status": "logged", **ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/recent")
def audit_recent(n: int = 5):
    try:
        _, get_recent_entries, _ = get_audit()
        return get_recent_entries(n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/stats")
def audit_stats():
    try:
        _, _, get_stats = get_audit()
        return get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve the HTML frontend ───────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("demo.html")

@app.get("/{filepath:path}")
def static_file(filepath: str):
    if os.path.isfile(filepath) and not filepath.startswith("."):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Not found")
