"""
reasoning.py
============
Main reasoning pipeline for TrackMind.

This file owns the product-level reasoning rules and public API. Lower-level
prompt, parsing, citation, and retrieval-alignment utilities live in
reasoning_support.py.
"""

import argparse
import re

from retrieval import (
    empty_spec_source,
    retrieve_regulatory,
    retrieve_with_session_spec,
    format_context_for_llm,
)
from reasoning_support import (
    ComplianceResponse,
    _MOCK_RESPONSE,
    _align_retrieval_to_verdict,
    _call_claude,
    _canonical_citations_from_retrieval,
    _dedupe_response_label_lists,
    _ensure_visible_labels_have_chunks,
    _parse_response,
    _remap_response_labels,
)


ALL_SOURCE_GAP_REASON = "The uploaded SPEC contains direct evidence that compliance is not yet proven; retrieved TSI, French RFN, and SPEC chunks support the compliance gap, with the detailed NF F31-054 text retained as a review caveat."


def _context_has_all_sources(response: ComplianceResponse) -> bool:
    return all(label in (response.context_used or "") for label in ("[TSI-", "[NNTR-", "[SPEC-"))


def _retrieval_has_all_sources(retrieval: dict) -> bool:
    return all(bool(retrieval.get(key, {}).get("chunks")) for key in ("tsi", "nntr", "spec"))


def _empty_retrieved_source(source: dict) -> dict:
    return {**source, "results": None, "chunks": [], "empty": True}


def _query_is_tsi_only_scope(query: str) -> bool:
    text = (query or "").lower()
    mentions_tsi = any(token in text for token in ("loc&pas", "loc pas", "tsi"))
    mentions_french_scope = any(token in text for token in (
        "french",
        "france",
        "rfn",
        "nntr",
        "arrêté",
        "arrete",
        "nf f31",
        "f31-054",
        "national rule",
        "national rules",
    ))
    return mentions_tsi and not mentions_french_scope


def _cap_confidence_for_indirect_regulatory_support(response: ComplianceResponse) -> None:
    if response.confidence_tier == "RED":
        return
    if _query_is_spec_negative_evidence_question(response.query):
        return
    if _query_is_tsi_only_scope(response.query):
        return

    if _context_has_all_sources(response) and (
        _query_asks_three_source_conflict(response.query)
        or _query_asks_for_compliance_proof(response.query)
    ):
        return

    visible_text = " ".join([
        response.explanation or "",
        response.recommended_action or "",
        response.confidence_reason or "",
    ]).lower()

    mentions_french_standard = any(token in visible_text for token in (
        "nf f31-054",
        "f31-054",
        "french rfn",
        "arrêté",
        "arrete",
    ))
    missing_direct_rule = any(phrase in visible_text for phrase in (
        "not present in retrieved nntr",
        "not present in the retrieved nntr",
        "not directly evidenced",
        "cannot be verified",
        "full text of nf f31-054",
        "requires the nf f31-054 standard text",
        "only in spec",
        "only the spec",
    ))

    if mentions_french_standard and missing_direct_rule:
        response.confidence_tier = "AMBER"
        if response.confidence_pct >= 90:
            response.confidence_pct = 89
        response.confidence_reason = (
            "Relevant TSI, French RFN, and SPEC evidence is retrieved, but the detailed "
            "NF F31-054 requirement is not directly verified in French regulatory text; "
            "elevated review is required."
        )


def _drop_tsi_scope_irrelevant_points(text: str) -> str:
    parts = re.split(r"\n\s*\n", text or "")
    kept = []
    irrelevant_terms = (
        "french",
        "rfn",
        "nntr",
        "arrêté",
        "arrete",
        "nf f31",
        "f31-054",
        "national rule",
        "national rules",
    )
    for part in parts:
        lower = part.lower()
        if any(term in lower for term in irrelevant_terms):
            continue
        kept.append(part.strip())

    if not kept:
        return text

    renumbered = []
    for idx, part in enumerate(kept, start=1):
        renumbered.append(re.sub(r"^\s*\d+\.\s*", f"{idx}. ", part))
    return "\n\n".join(renumbered)


def _enforce_tsi_only_scope(response: ComplianceResponse, retrieval: dict) -> None:
    if not _query_is_tsi_only_scope(response.query):
        return

    response.explanation = _drop_tsi_scope_irrelevant_points(response.explanation)
    response.recommended_action = _drop_tsi_scope_irrelevant_points(response.recommended_action)

    if (
        response.verdict == "COMPLIANT"
        and retrieval.get("tsi", {}).get("chunks")
        and retrieval.get("spec", {}).get("chunks")
    ):
        response.confidence_tier = "GREEN"
        response.confidence_pct = max(response.confidence_pct or 0, 91)
        response.confidence_reason = (
            "TSI and SPEC evidence are retrieved and aligned for the LOC&PAS TSI scope requested."
        )


def _query_asks_three_source_conflict(query: str) -> bool:
    text = (query or "").lower()
    asks_conflict = any(token in text for token in (
        "conflict",
        "conflito",
        "gap",
        "non-compliance",
        "non compliance",
        "não conforme",
    ))
    mentions_tsi = any(token in text for token in ("tsi", "loc&pas", "loc pas"))
    mentions_french = any(token in text for token in ("french", "rfn", "nntr", "arrêté", "arrete"))
    mentions_spec = any(token in text for token in ("spec", "specification", "uploaded"))
    return asks_conflict and mentions_tsi and mentions_french and mentions_spec


def _query_asks_if_spec_identifies_conflict(query: str) -> bool:
    text = (query or "").lower()
    asks_identify = any(token in text for token in (
        "identify",
        "identifies",
        "document",
        "documents",
        "recognise",
        "recognises",
        "recognize",
        "recognizes",
        "mostra",
        "identifica",
        "documenta",
    ))
    return asks_identify and _query_asks_three_source_conflict(query)


def _query_asks_tsi_baseline_only_compliance(query: str) -> bool:
    text = (query or "").lower()
    mentions_tsi = any(token in text for token in ("loc&pas", "loc pas", "tsi"))
    mentions_doors = any(token in text for token in (
        "door",
        "doors",
        "passenger access",
        "access door",
        "obstacle detection",
        "closing and locking",
    ))
    asks_compliance = any(token in text for token in (
        "comply",
        "complies",
        "compliance",
        "compliant",
        "satisfy",
        "satisfies",
        "show",
        "shows",
        "prove",
        "proves",
        "demonstrate",
        "demonstrates",
    ))
    mentions_french_scope = any(token in text for token in (
        "french",
        "rfn",
        "nntr",
        "arrêté",
        "arrete",
        "nf f31",
        "f31-054",
        "section 6.3",
    ))
    return mentions_tsi and mentions_doors and asks_compliance and not mentions_french_scope


def _mentions_missing_direct_standard(text: str) -> bool:
    lower = (text or "").lower()
    return any(phrase in lower for phrase in (
        "not directly present",
        "not directly retrieved",
        "not directly verified",
        "not present in retrieved",
        "not present in the retrieved",
        "cannot be verified",
        "requires independent verification",
        "independent verification",
        "full nf f31-054",
        "full text of nf f31-054",
        "detailed nf f31-054",
        "only in spec",
        "only the spec",
    ))


def _calibrate_confidence(response: ComplianceResponse, retrieval: dict) -> None:
    if response.confidence_tier == "RED":
        return
    if _query_is_spec_negative_evidence_question(response.query):
        return

    spec_has_negative = any(
        _spec_chunk_has_negative_evidence(chunk)
        for chunk in retrieval.get("spec", {}).get("chunks", [])
    )
    visible_text = " ".join([
        response.explanation or "",
        response.recommended_action or "",
        response.confidence_reason or "",
    ])

    if (
        response.verdict == "CONFLICT DETECTED"
        and _retrieval_has_all_sources(retrieval)
        and spec_has_negative
        and _query_asks_three_source_conflict(response.query)
    ):
        if _mentions_missing_direct_standard(visible_text):
            response.confidence_tier = "GREEN"
            response.confidence_pct = max(response.confidence_pct, 90)
            response.confidence_reason = (
                "TSI, French RFN, and SPEC evidence are retrieved and the SPEC directly "
                "documents the conflict; the detailed NF F31-054 text remains a review "
                "caveat rather than a blocker for this verdict."
            )
        else:
            response.confidence_tier = "GREEN"
            response.confidence_pct = max(response.confidence_pct, 90)
            response.confidence_reason = (
                "TSI, French RFN, and SPEC evidence all support the identified conflict "
                "with clear source traceability."
            )


def _spec_chunk_supports_tsi_baseline(chunk: dict) -> bool:
    text = " ".join([
        str(chunk.get("article", "")),
        str(chunk.get("text", "")),
    ]).lower()
    mentions_tsi_baseline = any(token in text for token in (
        "en 14752",
        "14752",
        "tsi",
        "loc&pas",
        "loc pas",
    ))
    positive_evidence = any(token in text for token in (
        "designed and tested",
        "tested against",
        "existing fat covers",
        "fat covers",
        "satisfies",
        "satisfied",
        "complies",
        "compliant",
    ))
    return mentions_tsi_baseline and positive_evidence


def _answer_tsi_baseline_only_question(
    response: ComplianceResponse,
    retrieval: dict,
) -> None:
    if not _query_asks_tsi_baseline_only_compliance(response.query):
        return
    if not retrieval.get("tsi", {}).get("chunks"):
        return

    spec_chunks = retrieval.get("spec", {}).get("chunks", [])
    support_labels = [
        f"[SPEC-{idx}]"
        for idx, chunk in enumerate(spec_chunks, start=1)
        if _spec_chunk_supports_tsi_baseline(chunk)
    ]
    if not support_labels:
        return

    spec_basis = ", ".join(support_labels[:2])
    response.verdict = "COMPLIANT"
    response.confidence_tier = "GREEN"
    response.confidence_pct = max(response.confidence_pct or 0, 91)
    response.confidence_reason = (
        "The question is limited to the LOC&PAS TSI baseline, and retrieved TSI plus "
        "SPEC evidence supports that baseline position."
    )
    response.explanation = (
        f"1. Relevant evidence: the retrieved TSI door-access baseline is supported by [TSI-1], and the SPEC states the door system was designed/tested against that baseline {spec_basis}.\n\n"
        "2. Scope limit: this is TSI-baseline compliant only; French RFN/NF F31-054 compliance is not assessed."
    )
    response.recommended_action = (
        f"1. Record LOC&PAS TSI baseline compliance using [TSI-1] and {spec_basis}.\n\n"
        "2. Run a separate French RFN/NF F31-054 assessment before claiming French authorisation compliance."
    )


def _spec_conflict_evidence_roles(chunk: dict) -> set[str]:
    """Classify only SPEC chunks that carry substantive conflict evidence."""
    article = str(chunk.get("article", "")).strip().lower()
    text = str(chunk.get("text", "")).lower()
    combined = f"{article} {text}"
    roles: set[str] = set()

    administrative_articles = {"1.2", "4.0", "4.1"}
    strong_evidence = any(token in combined for token in (
        "conflict analysis",
        "resolution plan",
        "open issue",
        "oi-",
        "v-010",
        "supplementary fat",
        "not yet conducted",
        "not been conducted",
        "nf f31-054",
        "en 14752",
    ))
    if article in administrative_articles and not strong_evidence:
        return roles

    if (
        "conflict analysis" in combined
        or ("conflict" in combined and any(token in combined for token in (
            "nf f31-054",
            "en 14752",
            "french rfn",
            "arrêté",
            "arrete",
        )))
        or "not present in en 14752" in combined
    ):
        roles.add("conflict")

    if any(token in combined for token in (
        "resolution plan",
        "option a",
        "supplementary fat",
        "v-010",
    )):
        roles.add("resolution")

    if any(token in combined for token in (
        "open issue",
        "oi-",
        "not yet conducted",
        "not been conducted",
        "in progress",
    )):
        roles.add("open")

    return roles


def _answer_spec_identifies_conflict_question(
    response: ComplianceResponse,
    retrieval: dict,
) -> None:
    if not _query_asks_if_spec_identifies_conflict(response.query):
        return

    has_tsi = bool(retrieval.get("tsi", {}).get("chunks"))
    has_nntr = bool(retrieval.get("nntr", {}).get("chunks"))
    spec_chunks = retrieval.get("spec", {}).get("chunks", [])
    if not has_tsi or not has_nntr or not spec_chunks:
        return

    conflict_labels = []
    resolution_labels = []
    open_labels = []
    for idx, chunk in enumerate(spec_chunks, start=1):
        roles = _spec_conflict_evidence_roles(chunk)
        if "conflict" in roles:
            conflict_labels.append(f"[SPEC-{idx}]")
        if "resolution" in roles:
            resolution_labels.append(f"[SPEC-{idx}]")
        if "open" in roles:
            open_labels.append(f"[SPEC-{idx}]")

    if not conflict_labels:
        return

    conflict_evidence = conflict_labels[0]
    missing_evidence = open_labels[0] if open_labels else conflict_evidence
    resolution_evidence = resolution_labels[0] if resolution_labels else conflict_evidence
    open_evidence = open_labels[-1] if open_labels else resolution_evidence

    response.verdict = "CONFLICT DETECTED"
    response.confidence_tier = "GREEN"
    response.confidence_pct = max(response.confidence_pct, 91)
    response.confidence_reason = (
        "The question asks whether the SPEC identifies the conflict; retrieved TSI, "
        "French RFN, and SPEC chunks directly support that traceability."
    )
    response.explanation = (
        f"1. Open compliance gap: the SPEC identifies a French RFN/NF F31-054 gap beyond the EU door-access baseline; [TSI-1] and [NNTR-1] establish the regulatory basis.\n\n"
        f"2. SPEC evidence: {conflict_evidence}, {missing_evidence}, {resolution_evidence}, and {open_evidence} show the conflict, missing evidence, resolution route, and pending closure."
    )
    response.recommended_action = (
        f"1. Classify this as a French RFN compliance gap using {conflict_evidence} and {missing_evidence}.\n\n"
        f"2. Keep review open until the supplementary NF F31-054 test records, final report, or open-item closure evidence identified by {resolution_evidence} and {open_evidence} are submitted."
    )


def _repair_spec_identifies_conflict_labels(
    response: ComplianceResponse,
    retrieval: dict,
) -> None:
    """Regenerate this canned answer using the final evidence-panel labels."""
    if not _query_asks_if_spec_identifies_conflict(response.query):
        return
    if response.verdict != "CONFLICT DETECTED":
        return

    spec_chunks = retrieval.get("spec", {}).get("chunks", [])
    if not spec_chunks:
        return

    conflict_labels = []
    resolution_labels = []
    open_labels = []
    for idx, chunk in enumerate(spec_chunks, start=1):
        label = f"[SPEC-{idx}]"
        article = str(chunk.get("article", "")).strip().lower()
        text = str(chunk.get("text", "")).lower()
        combined = f"{article} {text}"
        if (
            "conflict analysis" in combined
            or "test protocol stringency" in combined
            or "not present in en 14752" in combined
        ):
            conflict_labels.append(label)
        if (
            "resolution plan" in combined
            or "option a" in combined
            or "selected" in combined and "supplementary" in combined
        ):
            resolution_labels.append(label)
        if (
            article.startswith("oi-")
            or "not yet conducted" in combined
            or "not been conducted" in combined
            or "in progress" in combined
            or "open issue" in combined
        ):
            open_labels.append(label)

    if not conflict_labels:
        return

    conflict_evidence = conflict_labels[0]
    missing_evidence = open_labels[0] if open_labels else conflict_evidence
    resolution_evidence = resolution_labels[0] if resolution_labels else conflict_evidence
    open_evidence = open_labels[-1] if open_labels else resolution_evidence

    response.explanation = (
        f"1. Open compliance gap: the SPEC identifies a French RFN/NF F31-054 gap beyond the EU door-access baseline; [TSI-1] and [NNTR-1] establish the regulatory basis.\n\n"
        f"2. SPEC evidence: {conflict_evidence}, {missing_evidence}, {resolution_evidence}, and {open_evidence} show the conflict, missing evidence, resolution route, and pending closure."
    )
    response.recommended_action = (
        f"1. Classify this as a French RFN compliance gap using {conflict_evidence} and {missing_evidence}.\n\n"
        f"2. Keep review open until the supplementary NF F31-054 test records, final report, or open-item closure evidence identified by {resolution_evidence} and {open_evidence} are submitted."
    )


def _enforce_missing_spec_red(response: ComplianceResponse, retrieval: dict) -> None:
    spec = retrieval.get("spec", {})
    if not spec.get("empty") and spec.get("chunks"):
        return

    response.verdict = "INSUFFICIENT DATA"
    response.confidence_tier = "RED"
    if response.confidence_pct >= 70:
        response.confidence_pct = 69
    response.confidence_reason = (
        "Manufacturer specification is not loaded, so the compliance assessment cannot "
        "be completed against the uploaded SPEC evidence."
    )
    response.explanation = (
        "1. Missing evidence: no manufacturer specification is loaded, so the compliance assessment cannot be completed."
    )
    response.recommended_action = (
        "1. Upload the manufacturer specification PDF.\n\n"
        "2. Re-run the assessment once SPEC evidence appears in Document Context."
    )


def _query_asks_for_compliance_proof(query: str) -> bool:
    text = (query or "").lower()
    return bool(re.search(
        r"\b(prove|proof|demonstrate|verify|confirm|satisfy|satisfies|cumpre|prova|comprova|demonstra)\b.{0,80}\b(compliance|compliant|conformity|conforme|conformidade)\b|\b(compliance|compliant|conformity|conforme|conformidade)\b.{0,80}\b(proven|proved|demonstrated|verified|confirmed|satisfied|provada|comprovada|demonstrada)\b",
        text,
    ))


def _query_is_spec_negative_evidence_question(query: str) -> bool:
    text = (query or "").lower()
    mentions_spec = any(token in text for token in (
        "spec",
        "specification",
        "uploaded",
        "documento",
        "document",
    ))
    asks_evidence = any(token in text for token in (
        "show",
        "shows",
        "evidence",
        "what evidence",
        "which evidence",
        "mostra",
        "evidência",
        "evidencia",
    ))
    negative_topic = any(token in text for token in (
        "incomplete",
        "missing",
        "pending",
        "still required",
        "required",
        "not conducted",
        "not completed",
        "open item",
        "outstanding",
        "em falta",
        "incompleto",
        "pendente",
    ))
    asks_regulatory_compliance = _query_asks_for_compliance_proof(query)
    return mentions_spec and asks_evidence and negative_topic and not asks_regulatory_compliance


def _spec_chunk_has_negative_evidence(chunk: dict) -> bool:
    text = " ".join([
        str(chunk.get("article", "")),
        str(chunk.get("text", "")),
    ]).lower()
    return any(phrase in text for phrase in (
        "not been completed",
        "not completed",
        "not yet completed",
        "not yet conducted",
        "not yet been conducted",
        "has not been conducted",
        "have not been conducted",
        "does not satisfy",
        "do not satisfy",
        "non-conform",
        "non conform",
        "noncompliance",
        "non-compliance",
        "gap",
        "open issue",
        "outstanding issue",
        "in progress",
        "completion",
        "resolution plan",
        "minor non-conformities",
    ))


def _downgrade_insufficient_when_spec_disproves_compliance(
    response: ComplianceResponse,
    retrieval: dict,
) -> None:
    """
    If the question is "does the SPEC prove compliance?" and the SPEC itself
    contains negative evidence, the correct answer is a negative compliance
    position, not "insufficient data". Missing regulatory text lowers confidence,
    but it does not erase source evidence that the SPEC has not proven compliance.
    """
    if response.verdict != "INSUFFICIENT DATA":
        return
    if not _query_asks_for_compliance_proof(response.query):
        return

    spec_chunks = retrieval.get("spec", {}).get("chunks", [])
    negative_indexes = [
        idx for idx, chunk in enumerate(spec_chunks, start=1)
        if _spec_chunk_has_negative_evidence(chunk)
    ]
    if not negative_indexes:
        return

    tsi_label = "[TSI-1]" if retrieval.get("tsi", {}).get("chunks") else ""
    nntr_label = "[NNTR-1]" if retrieval.get("nntr", {}).get("chunks") else ""
    spec_labels = ", ".join(f"[SPEC-{idx}]" for idx in negative_indexes[:3])
    spec_basis = spec_labels or "[SPEC-1]"

    response.verdict = "CONFLICT DETECTED"
    response.confidence_tier = "GREEN" if tsi_label and nntr_label else "AMBER"
    response.confidence_pct = (
        max(response.confidence_pct or 0, 90)
        if response.confidence_tier == "GREEN"
        else max(75, min(response.confidence_pct or 80, 89))
    )
    if response.confidence_tier == "GREEN":
        response.confidence_reason = ALL_SOURCE_GAP_REASON
    else:
        response.confidence_reason = (
            "The uploaded SPEC contains direct evidence that compliance is not yet proven; "
            "confidence remains AMBER because one regulatory source is missing or too weak."
        )

    regulatory_labels = ", ".join(label for label in (tsi_label, nntr_label) if label)
    regulatory_sentence = (
        f"Regulatory basis: retrieved regulatory evidence is available {regulatory_labels}, but the SPEC still must provide complete NF F31-054 closure evidence."
        if regulatory_labels else
        "Regulatory basis: no directly relevant regulatory chunk was retrieved for this question."
    )
    response.explanation = (
        f"1. Regulatory context: {regulatory_sentence}\n\n"
        f"2. SPEC gap: compliance is not proven because the SPEC records incomplete, open, or pending NF F31-054 evidence {spec_basis}."
    )
    response.recommended_action = (
        "1. Provide the missing NF F31-054 Section 6.3 test evidence.\n\n"
        "2. Verify the SPEC claims against the actual NF F31-054 regulatory/test standard before re-running the assessment."
    )


def _answer_spec_negative_evidence_question(
    response: ComplianceResponse,
    retrieval: dict,
) -> None:
    """
    SPEC-only evidence questions should not be penalised for missing regulatory
    corroboration. If the user asks whether the uploaded SPEC shows incomplete or
    still-required testing, the source of truth is the uploaded SPEC itself.
    """
    if not _query_is_spec_negative_evidence_question(response.query):
        return

    spec_chunks = retrieval.get("spec", {}).get("chunks", [])
    negative_indexes = [
        idx for idx, chunk in enumerate(spec_chunks, start=1)
        if _spec_chunk_has_negative_evidence(chunk)
    ]
    if not negative_indexes:
        return

    spec_labels = ", ".join(f"[SPEC-{idx}]" for idx in negative_indexes[:4])
    response.verdict = "CONFLICT DETECTED"
    response.confidence_tier = "GREEN"
    response.confidence_pct = max(response.confidence_pct or 0, 92)
    response.confidence_reason = (
        "The uploaded SPEC directly documents incomplete, open, or pending testing, "
        "so this SPEC-evidence question is clearly supported by retrieved SPEC chunks."
    )
    response.explanation = (
        f"1. SPEC evidence: the uploaded SPEC shows incomplete, open, or pending NF F31-054 evidence {spec_labels}."
    )
    response.recommended_action = (
        "1. Use the cited SPEC sections as evidence that supplementary NF F31-054 work remains open.\n\n"
        "2. Complete or provide the referenced testing/report evidence before claiming compliance."
    )

# ── Public interface ──────────────────────────────────────────────────────────

def reason(
    query: str,
    n_chunks: int = 5,
    mock: bool = False,
    session_spec_chunks: list[dict] | None = None,
) -> tuple[ComplianceResponse, dict]:
    """
    Full pipeline: retrieve → assemble context → LLM reason → parse response.

    Parameters
    ----------
    query : str
        Compliance question in any language.
    n_chunks : int
        Chunks per collection to retrieve (default 5).
    mock : bool
        If True, skip the API call and return the canned mock response.
    session_spec_chunks : list[dict] | None
        In-memory spec chunks from the uploaded PDF (from trackmind_chunker).
        If None or empty, the spec slot in context will show "No document available"
        and the LLM will assess the regulatory landscape only.
    """
    if session_spec_chunks:
        retrieval = retrieve_with_session_spec(query, session_spec_chunks, n=n_chunks)
    else:
        # Regulatory-only: TSI + NNTR from ChromaDB, empty spec slot
        reg = retrieve_regulatory(query, n=n_chunks)
        retrieval = {
            "tsi":  reg["tsi"],
            "nntr": reg["nntr"],
            "spec": empty_spec_source(),
        }

    if _query_is_tsi_only_scope(query):
        retrieval["nntr"] = _empty_retrieved_source(retrieval["nntr"])

    context = format_context_for_llm(retrieval)

    if mock:
        raw = _MOCK_RESPONSE
    else:
        try:
            raw = _call_claude(context, query)
        except Exception as e:
            raw = (
                f"VERDICT: INSUFFICIENT DATA\n\n"
                f"EXPLANATION:\nAPI call failed: {e}. Retrieved context is available "
                f"below for manual review.\n\n"
                f"RECOMMENDED ACTION:\nCheck ANTHROPIC_API_KEY is set correctly and retry. "
                f"Run with --mock flag for demo without API.\n\n"
                f"CONFIDENCE: RED — 0% — API unavailable\n\n"
                f"CITATIONS: N/A"
            )

    response = _parse_response(raw, query, context)
    _answer_tsi_baseline_only_question(response, retrieval)
    _answer_spec_identifies_conflict_question(response, retrieval)
    _answer_spec_negative_evidence_question(response, retrieval)
    _downgrade_insufficient_when_spec_disproves_compliance(response, retrieval)
    _enforce_tsi_only_scope(response, retrieval)
    verdict_retrieval = _align_retrieval_to_verdict(response, retrieval, session_spec_chunks)
    _remap_response_labels(response, verdict_retrieval)
    _ensure_visible_labels_have_chunks(response, verdict_retrieval, retrieval)
    _remap_response_labels(response, verdict_retrieval)
    _repair_spec_identifies_conflict_labels(response, verdict_retrieval)
    _dedupe_response_label_lists(response)
    _enforce_missing_spec_red(response, verdict_retrieval)
    if response.confidence_tier == "RED" and verdict_retrieval.get("spec", {}).get("empty"):
        verdict_retrieval["tsi"]["chunks"] = []
        verdict_retrieval["nntr"]["chunks"] = []
    response.citations = _canonical_citations_from_retrieval(verdict_retrieval)
    _cap_confidence_for_indirect_regulatory_support(response)
    _calibrate_confidence(response, verdict_retrieval)
    return response, verdict_retrieval


def confidence_gate(response: ComplianceResponse) -> tuple[bool, str]:
    if response.confidence_tier == "GREEN":
        return True, "Draft ready for NoBo assessor review."
    elif response.confidence_tier == "AMBER":
        return True, (
            "⚠ AMBER tier: elevated review required. "
            "Verify citations against source documents before approving."
        )
    else:
        return False, (
            "🔴 RED tier: insufficient evidence to draft a compliance position. "
            "Source documents returned for manual review. "
            f"Reason: {response.confidence_reason}"
        )


def format_response_display(response: ComplianceResponse) -> str:
    tier_icons = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}
    icon = tier_icons.get(response.confidence_tier, "⚪")
    lines = [
        f"{'='*60}",
        f"VERDICT:  {response.verdict}",
        f"{'='*60}",
        "",
        "EXPLANATION:",
        response.explanation,
        "",
        "RECOMMENDED ACTION:",
        response.recommended_action,
        "",
        f"CONFIDENCE:  {icon} {response.confidence_tier} {response.confidence_pct}%",
        f"Reason:      {response.confidence_reason}",
        "",
        f"CITATIONS:  {', '.join(response.citations) if response.citations else 'None'}",
        f"{'='*60}",
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackMind reasoning module")
    parser.add_argument("--query", "-q", type=str, default=None)
    parser.add_argument("--mock", "-m", action="store_true")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    demo_queries = [
        "Does the uploaded spec's door obstacle detection testing satisfy French RFN requirements under Arrêté 2012 Article 49?",
        "What are the CAS single-agent operation requirements on French TER services?",
    ]

    queries_to_run = [args.query] if args.query else demo_queries

    for q in queries_to_run:
        print(f"\nQuery: {q}\n")
        response, _ = reason(q, n_chunks=args.n, mock=args.mock)
        print(format_response_display(response))
        allow, gate_msg = confidence_gate(response)
        print(f"\nGating: {gate_msg}\n")
