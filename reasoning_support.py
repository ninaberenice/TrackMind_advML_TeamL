"""
reasoning_support.py
====================
Support code for TrackMind's LLM reasoning layer.

Contains the long system prompt, response parsing, citation remapping,
retrieval-panel alignment, mock response, and Claude API call.
"""

import re
import os
from dataclasses import dataclass
from retrieval import (
    retrieve_regulatory_article,
)

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a TSI compliance reasoning engine for EU railway certification.

You receive retrieved chunks from up to three document sources:

  - TSI:  LOC&PAS Commission Regulation (EU) 1302/2014 (consolidated to 2025) — English
  - NNTR: Arrêté du 19 mars 2012 fixant les objectifs, les méthodes, les indicateurs
          de sécurité et la réglementation technique applicable sur le réseau ferré national
          (French national rule) — French language
  - SPEC: Manufacturer's technical specification uploaded for this session — English
          (This is the document under assessment. Treat its claims as the manufacturer's
          stated position, to be validated against TSI and NNTR requirements.)

REGULATORY HIERARCHY (apply this in every response):

1. TSI PRIMACY: The LOC&PAS TSI is the EU baseline. Under Directive 2016/797 Article 4,
   the TSI takes precedence over national rules UNLESS the national rule covers a matter
   not addressed by the TSI (a "gap") or is explicitly notified to ERA as a national rule.

2. FRENCH NATIONAL RULE: The Arrêté du 19 mars 2012 Article 49 contains mandatory
   rolling stock requirements for operation on the French Réseau Ferré National (RFN).
   These are BINDING for French RFN authorisation (AMEC issuance by EPSF) even where
   they exceed or differ from the TSI.

3. NF F31-054: The French national standard referenced in Art. 49 for passenger door
   systems on CAS-operated (Conduite Agent Seul / single-agent-operated) trains.
   Treat NF F31-054 requirements as binding for French TER services.
   Key NF F31-054 requirements:
   - Obstacle detection: 5 height positions (250/500/900/1300/1600 mm), ≤1.5 J kinetic energy
   - CAS platform surveillance: visual cab confirmation required before door closure
   - CAS closure confirmation: mandatory two-step interlock sequence
   - CAS passenger alarm: door lock-open + active agent acknowledgement required
   - CAS re-closure dwell: minimum 5 s after obstacle detection reversal

4. EN 14752: Treat EN 14752 as relevant to passenger door systems only if it appears
   in the retrieved TSI/SPEC chunks. Do not invent or reuse a LOC&PAS TSI article number
   from memory; cite only the retrieved chunk label and article/section shown in context.

ASSESSMENT RULES (non-negotiable):
- The retrieved chunks are the only citable evidence. The rules in this system prompt
  are assessment guidance, not source evidence.
- Every article, section, standard, and SPEC claim mentioned in EXPLANATION,
  RECOMMENDED ACTION, or CITATIONS must be supported by an explicit retrieved chunk
  label such as [TSI-1], [NNTR-2], or [SPEC-1].
- A SPEC chunk describing what a regulation requires is only evidence of the
  manufacturer's/specification claim. It is not regulatory evidence. TSI claims need
  [TSI-n] support; French RFN/Arrêté/NF F31-054 claims need [NNTR-n] support.
- If a French RFN/NF F31-054 detail is only described in SPEC and the matching
  NNTR/regulatory chunk establishes the applicable French RFN/NF F31-054 basis
  but not the detailed test protocol, confidence may still be GREEN when the
  question is about the SPEC's documented gap/position and TSI, NNTR, and SPEC
  chunks are all retrieved. Make the limitation explicit in the explanation.
- If SPEC cites a TSI or NNTR article, verify that article against the matching TSI/NNTR
  chunk. If the regulatory chunk does not support the SPEC's description, state that
  the SPEC regulatory reference appears inconsistent or unsupported.
- If the user names a specific TSI/NNTR article or section and that exact
  article/section appears in the retrieved context under a chunk label, treat it as
  retrieved and cite that label. Do not say the article was not retrieved.
- If you know a requirement from the guidance but the matching retrieved chunk is not
  present, do not cite it as evidence. Mark the answer INSUFFICIENT DATA or explain
  that the relevant source chunk was not retrieved.
- If the user asks whether the uploaded SPEC proves compliance, and retrieved SPEC
  chunks explicitly state that required testing is incomplete, not yet conducted,
  open, outstanding, or non-conforming, do NOT answer INSUFFICIENT DATA solely
  because the full regulatory standard text is missing. Answer that the SPEC does
  not prove compliance / shows a gap, and use AMBER if the regulatory detail still
  requires independent source verification.
- If the user asks only what the uploaded SPEC says or what evidence in the SPEC
  shows incomplete/missing/pending testing, answer from SPEC chunks only. Do not
  add generic TSI/NNTR anchors unless the question asks for regulatory compliance
  against those sources.
- If the user asks only about LOC&PAS TSI / TSI requirements and does not mention
  French RFN, NNTR, Arrêté, NF F31-054, or national rules, assess only the TSI scope.
  Do not discuss French RFN enhancements or actions, even if the SPEC mentions them.
- Cite the specific article/section number for EVERY factual claim
- Compare the SPEC claims directly against TSI and NNTR requirements on each parameter
- If the same parameter is covered by both TSI and NNTR with different requirements,
  identify this as a CONFLICT and explain which takes precedence and why
- If the SPEC satisfies both TSI and NNTR on a parameter, state COMPLIANT for that parameter
- If a relevant article is missing from the SPEC or regulatory sources, say so explicitly
- If the SPEC is not uploaded (source shows "No document available"), assess only the
  regulatory landscape and note that a spec is required for full compliance assessment
- Never guess. If evidence is insufficient, use RED tier and explain the gap
- Keep reasoning concise — the NoBo assessor reads dozens of these per day
- If the NNTR chunk is in French, reason from it directly — do not refuse

OUTPUT FORMAT (strict — machine-parsed):

VERDICT: [COMPLIANT / CONFLICT DETECTED / INSUFFICIENT DATA]

EXPLANATION:
Use a concise numbered list with 1-3 points. Decide the order, point headings, and
level of detail based on what is most relevant to the user's question and the
verdict. Do not force fixed headings such as "TSI baseline", "French RFN delta",
or "SPEC finding". State only decision-critical facts and cite supporting chunk
labels e.g. [TSI-1], [NNTR-2], [SPEC-1].
When citing multiple chunks, write each label separately, e.g. [SPEC-1], [SPEC-4].
Never write grouped labels like [SPEC-1, SPEC-4].
Do not cite a source label only to say that a detail is absent. Cite the article that
establishes the applicable requirement, and explain any missing detail without adding
irrelevant negative-evidence chunks.
Keep the full EXPLANATION concise, preferably 60-110 words. Do not quote long source
text. Do not mention non-decision-critical background.

RECOMMENDED ACTION:
Use a concise numbered list with only the action(s) needed for this verdict, in
the order that matters operationally. Be specific: name the exact test, report,
traceability matrix, closure record, or source document when relevant. Avoid
generic process advice unless it is the actual next step.

CONFIDENCE: [GREEN / AMBER / RED] — [XX%] — [one-sentence reason for this tier]

CITATIONS: [comma-separated list of article/section references used, each including
the supporting chunk label and the article/section exactly as shown in that chunk]

CONFIDENCE TIER DEFINITIONS:
  GREEN  (>=90%): All relevant articles found in retrieved context. Position clearly
                 supported by source text. Safe for NoBo review.
  AMBER (70-89%): Relevant articles found but position involves substantial
                  inference across sources, or one source is missing/incomplete.
                  Use high AMBER (85-89%) when source coverage is mostly strong
                  but a decision-critical detail still needs independent
                  verification. Use GREEN when TSI, NNTR, and SPEC evidence are
                  all retrieved and the remaining limitation is a clearly stated
                  caveat rather than the basis for the verdict.
  RED   (<70%):  Key source documents missing, or question outside ingested scope.
                 Do NOT draft — return source chunks and explain the gap.
"""


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class ComplianceResponse:
    verdict:            str
    explanation:        str
    recommended_action: str
    confidence_tier:    str
    confidence_pct:     int
    confidence_reason:  str
    citations:          list[str]
    raw_response:       str
    context_used:       str
    query:              str


def _parse_response(raw: str, query: str, context: str) -> ComplianceResponse:
    """Extract structured fields from the LLM's raw text output."""

    def _extract(label: str, text: str) -> str:
        pattern = rf'{label}:\s*(.*?)(?=\n[A-Z ]+:|$)'
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    verdict       = _extract("VERDICT", raw).split("\n")[0].strip()
    explanation   = _extract("EXPLANATION", raw)
    rec_action    = _extract("RECOMMENDED ACTION", raw)
    citations_raw = _extract("CITATIONS", raw)
    citations     = [c.strip() for c in citations_raw.split(",") if c.strip()]

    conf_raw = _extract("CONFIDENCE", raw)
    conf_tier, conf_pct, conf_reason = "RED", 0, conf_raw
    conf_match = re.search(
        r'(GREEN|AMBER|RED)\s*[—\-–]\s*(\d+)%?\s*[—\-–]\s*(.*)',
        conf_raw, re.IGNORECASE
    )
    if conf_match:
        conf_tier   = conf_match.group(1).upper()
        conf_pct    = int(conf_match.group(2))
        conf_reason = conf_match.group(3).strip()

    verdict_upper = verdict.upper()
    if "CONFLICT" in verdict_upper:
        verdict = "CONFLICT DETECTED"
    elif "COMPLIANT" in verdict_upper:
        verdict = "COMPLIANT"
    elif "INSUFFICIENT" in verdict_upper:
        verdict = "INSUFFICIENT DATA"

    return ComplianceResponse(
        verdict=verdict,
        explanation=explanation,
        recommended_action=rec_action,
        confidence_tier=conf_tier,
        confidence_pct=conf_pct,
        confidence_reason=conf_reason,
        citations=citations,
        raw_response=raw,
        context_used=context,
        query=query,
    )


def _article_refs_from_citation(text: str, source_key: str) -> list[str]:
    refs = []
    if source_key == "tsi":
        refs.extend(re.findall(r"(?:\[TSI-\d+\]|LOC&PAS\s+TSI|LOC\s+PAS\s+TSI|TSI).{0,100}?(?:art\.?|section|sec\.?)\s*(\d+(?:\.\d+){2,})", text, flags=re.IGNORECASE | re.DOTALL))
        refs.extend(re.findall(r"(?:\[TSI-\d+\]|LOC&PAS\s+TSI|LOC\s+PAS\s+TSI|TSI).{0,60}?\b(Article\s+\d+)\b(?!\.)", text, flags=re.IGNORECASE | re.DOTALL))
    elif source_key == "nntr":
        refs.extend(f"Art. {num}" for num in re.findall(r"(?:\[NNTR-\d+\]|Arrêté|Arrete|French\s+(?:RFN|national)|RFN\s+rule|NNTR).{0,120}?\bArt\.?\s*(\d+(?:er|ère|re|nd)?)\b", text, flags=re.IGNORECASE | re.DOTALL))
        refs.extend(f"Art. {num}" for num in re.findall(r"(?:\[NNTR-\d+\]|Arrêté|Arrete|French\s+(?:RFN|national)|RFN\s+rule|NNTR).{0,120}?\bArticle\s+(\d+(?:er|ère|re|nd)?)\b", text, flags=re.IGNORECASE | re.DOTALL))
    elif source_key == "spec":
        refs.extend(re.findall(r"\b(OI-\d+)\b", text, flags=re.IGNORECASE))
        refs.extend(re.findall(r"\b(V-\d+)\b", text, flags=re.IGNORECASE))
        refs.extend(re.findall(r"(?:\[SPEC-\d+\]|SPEC|uploaded specification).{0,80}?(?:section|sec\.?)?\s*(\d+(?:\.\d+){1,})", text, flags=re.IGNORECASE | re.DOTALL))

    unique = []
    for ref in refs:
        ref = re.sub(r"\s+", " ", ref).strip().rstrip(".")
        if ref and ref not in unique:
            unique.append(ref)
    return unique


def _session_spec_article_chunks(article_refs: list[str], session_spec_chunks: list[dict] | None) -> list[dict]:
    if not session_spec_chunks:
        return []

    wanted = {ref.lower() for ref in article_refs}
    chunks = []
    for chunk in session_spec_chunks:
        article = str(chunk.get("metadata", {}).get("article", "")).lower()
        if article not in wanted:
            continue
        chunks.append({
            "text":     chunk["text"],
            "article":  chunk["metadata"].get("article", "unknown"),
            "language": chunk["metadata"].get("language", "en"),
            "doc_type": chunk["metadata"].get("doc_type", "SPEC"),
            "distance": None,
            "source":   chunk["metadata"].get("source_file", "uploaded"),
        })
    return chunks


def _chunk_key(chunk: dict) -> tuple[str, str]:
    return (str(chunk.get("article", "")).lower(), str(chunk.get("text", ""))[:160].lower())


def _append_unique(target: list[dict], chunk: dict) -> None:
    key = _chunk_key(chunk)
    if all(_chunk_key(existing) != key for existing in target):
        target.append(chunk)


def _iter_source_label_refs(text: str):
    for source, idx in re.findall(r"\b(TSI|NNTR|SPEC)-(\d+)\b", text or "", flags=re.IGNORECASE):
        yield source.upper(), idx


def _label_context_is_negative_only(text: str, label: str) -> bool:
    match = re.search(r"\[?\b" + re.escape(label) + r"\b\]?", text or "", flags=re.IGNORECASE)
    if not match:
        return False

    start = max(
        text.rfind(".", 0, match.start()),
        text.rfind("\n", 0, match.start()),
        text.rfind(";", 0, match.start()),
    ) + 1
    end_candidates = [
        pos for pos in (
            text.find(".", match.end()),
            text.find("\n", match.end()),
            text.find(";", match.end()),
        )
        if pos != -1
    ]
    end = min(end_candidates) if end_candidates else len(text)
    context = text[start:end].lower()

    negative = any(phrase in context for phrase in (
        "but do not",
        "but does not",
        "do not address",
        "does not address",
        "do not contain",
        "does not contain",
        "does not show",
        "do not cover",
        "does not cover",
        "not present",
        "not retrieved",
        "missing from",
        "absent from",
        "source missing",
        "cannot be verified",
        "cannot be performed",
        "cannot proceed",
        "no ",
    ))
    positive = any(phrase in context for phrase in (
        "requires",
        "mandates",
        "imposes",
        "establishes",
        "confirms",
        "states",
        "supports",
        "documents",
        "identifies",
    ))
    hard_negative = any(phrase in context for phrase in (
        "but do not",
        "but does not",
        "do not address",
        "does not address",
        "do not contain",
        "does not contain",
        "not present",
        "missing from",
        "cannot be verified",
        "cannot proceed",
    ))
    return hard_negative or (negative and not positive)


def _append_labeled_unique(target: list[dict], chunk: dict, label: str) -> None:
    labeled = {**chunk, "_source_labels": [label]}
    key = _chunk_key(labeled)
    for existing in target:
        if _chunk_key(existing) == key:
            labels = existing.setdefault("_source_labels", [])
            if label not in labels:
                labels.append(label)
            return
    target.append(labeled)


def _remap_response_labels(response: ComplianceResponse, retrieval: dict) -> None:
    replacements = {}
    for key in ("tsi", "nntr", "spec"):
        prefix = key.upper()
        for idx, chunk in enumerate(retrieval.get(key, {}).get("chunks", []), start=1):
            for original in chunk.get("_source_labels", []):
                replacements[original.strip("[]").upper()] = f"[{prefix}-{idx}]"

        used_labels = sorted(
            {
                int(idx)
                for source, idx in _iter_source_label_refs(" ".join([
                    response.explanation or "",
                    response.recommended_action or "",
                    response.confidence_reason or "",
                    response.raw_response or "",
                ]))
                if source == prefix
            }
        )
        if len(used_labels) == len(retrieval.get(key, {}).get("chunks", [])):
            for idx, old_num in enumerate(used_labels, start=1):
                replacements.setdefault(f"{prefix}-{old_num}", f"[{prefix}-{idx}]")

    if not replacements:
        return

    pattern = re.compile(
        r"\[?\b(" + "|".join(re.escape(label) for label in sorted(replacements, key=len, reverse=True)) + r")\b\]?",
        re.IGNORECASE,
    )

    def replace(text: str) -> str:
        return pattern.sub(lambda match: replacements[match.group(1).upper()], text or "")

    response.explanation = replace(response.explanation)
    response.recommended_action = replace(response.recommended_action)
    response.confidence_reason = replace(response.confidence_reason)
    response.raw_response = replace(response.raw_response)


def _dedupe_inline_label_lists(text: str) -> str:
    label = r"\[(?:TSI|NNTR|SPEC)-\d+\]"
    pattern = re.compile(rf"{label}(?:\s*,\s*{label})+", flags=re.IGNORECASE)

    def repl(match: re.Match) -> str:
        seen = set()
        labels = []
        for item in re.findall(label, match.group(0), flags=re.IGNORECASE):
            key = item.upper()
            if key in seen:
                continue
            seen.add(key)
            labels.append(item.upper())
        return ", ".join(labels)

    return pattern.sub(repl, text or "")


def _dedupe_response_label_lists(response: ComplianceResponse) -> None:
    response.explanation = _dedupe_inline_label_lists(response.explanation)
    response.recommended_action = _dedupe_inline_label_lists(response.recommended_action)
    response.confidence_reason = _dedupe_inline_label_lists(response.confidence_reason)
    response.raw_response = _dedupe_inline_label_lists(response.raw_response)


def _ensure_visible_labels_have_chunks(
    response: ComplianceResponse,
    aligned: dict,
    original: dict,
) -> None:
    visible_text = " ".join([
        response.explanation or "",
        response.recommended_action or "",
        response.confidence_reason or "",
    ])
    for source, idx in _iter_source_label_refs(visible_text):
        key = source.lower()
        pos = int(idx) - 1
        if pos < len(aligned.get(key, {}).get("chunks", [])):
            continue
        chunks = original.get(key, {}).get("chunks", [])
        if 0 <= pos < len(chunks):
            _append_labeled_unique(aligned[key]["chunks"], chunks[pos], f"[{source}-{idx}]")


def _align_retrieval_to_verdict(
    response: ComplianceResponse,
    retrieval: dict,
    session_spec_chunks: list[dict] | None,
) -> dict:
    """
    Rebuild the frontend evidence panel from the verdict, not from raw semantic rank.

    The LLM first reasons over retrieved context. After that, we inspect the exact
    chunk labels and citations it used, then surface those chunks in Document context.
    """
    aligned = {}
    for key in ("tsi", "nntr", "spec"):
        src = retrieval[key]
        aligned[key] = {**src, "chunks": [], "empty": src.get("empty", False)}

    # The evidence panel should show the evidence actually used in the visible
    # verdict, not every citation the model happened to list. The raw CITATIONS
    # field can contain extra nearby articles, so use only parsed visible fields
    # as the source of truth.
    verdict_refs = " ".join([
        response.explanation or "",
        response.recommended_action or "",
        response.confidence_reason or "",
    ])
    # Direct [TSI-1] / [NNTR-2] / [SPEC-3] references are the strongest signal:
    # they point to the exact context items the model used. Keep their original
    # labels so we can remap the visible verdict labels after filtering.
    for source, idx in _iter_source_label_refs(verdict_refs):
        key = source.lower()
        label = f"{source}-{idx}"
        if key in ("tsi", "nntr") and _label_context_is_negative_only(verdict_refs, label):
            continue
        chunks = retrieval.get(key, {}).get("chunks", [])
        pos = int(idx) - 1
        if 0 <= pos < len(chunks):
            _append_labeled_unique(aligned[key]["chunks"], chunks[pos], f"[{label}]")

    # Honor article references explicitly requested by the user. Do not expand the
    # evidence panel from incidental article mentions in the generated verdict:
    # visible [TSI-1] / [SPEC-1] labels are the source of truth there.
    for key in ("tsi", "nntr"):
        for article in _article_refs_from_citation(response.query, key):
            for chunk in retrieve_regulatory_article(key, article, n=2):
                _append_unique(aligned[key]["chunks"], chunk)

    for chunk in _session_spec_article_chunks(_article_refs_from_citation(response.query, "spec"), session_spec_chunks):
        _append_unique(aligned["spec"]["chunks"], chunk)

    for key in ("tsi", "nntr", "spec"):
        aligned[key]["empty"] = retrieval.get(key, {}).get("empty", False)

    return aligned


def _canonical_citations_from_retrieval(retrieval: dict) -> list[str]:
    labels = {
        "tsi": "LOC&PAS TSI",
        "nntr": "Arrêté du 19 mars 2012",
        "spec": "Uploaded specification",
    }
    citations = []
    for key in ("tsi", "nntr", "spec"):
        seen = set()
        for chunk in retrieval.get(key, {}).get("chunks", []):
            article = str(chunk.get("article", "")).strip()
            if not article or article.lower() == "unknown" or article.lower() in seen:
                continue
            seen.add(article.lower())
            citations.append(f"{labels[key]} {article}")
    return citations


# ── Mock response ─────────────────────────────────────────────────────────────

_MOCK_RESPONSE = """VERDICT: CONFLICT DETECTED

EXPLANATION:
The retrieved TSI and French RFN evidence show a stricter French door-system delta beyond the EU baseline [TSI-1], [NNTR-1]. The uploaded spec does not close that delta because the supplementary French obstacle-detection evidence remains missing [SPEC-1].

RECOMMENDED ACTION:
Commission the supplementary NF F31-054 evidence identified by the SPEC and keep the French RFN compliance item open until the completed test report is available.

CONFIDENCE: GREEN — 92% — Both conflicts are directly documented in retrieved NNTR Art. 49 and SPEC chunks. No inference required.

CITATIONS: [TSI-1] retrieved LOC&PAS TSI door-access section, [NNTR-1] Arrêté 2012 Art. 49, [SPEC-1] uploaded specification obstacle-detection section
"""


# ── Claude API call ───────────────────────────────────────────────────────────

def _call_claude(context: str, query: str) -> str:
    """
    Call Claude via the Anthropic REST API.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 2000,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"RETRIEVED CONTEXT:\n{context}\n\n"
                    f"COMPLIANCE QUERY:\n{query}"
                )
            }
        ],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["content"][0]["text"].strip()
