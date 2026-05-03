"""
reasoning.py
============
LLM reasoning layer for TrackMind.

Takes retrieved context from all three document collections and produces
a structured compliance position with confidence scoring (GREEN/AMBER/RED).

The system prompt encodes the regulatory hierarchy:
  - EU TSI primacy (Directive 2016/797 Art. 4)
  - French national rule (Arrêté du 19 mars 2012, Art. 49) as binding
    for RFN operation even where it exceeds the TSI
  - NF F31-054 as the French national standard referenced in Art. 49

Primary LLM: Google Gemini 1.5 Flash (free tier, no credit card required).
  Get a free API key at: aistudio.google.com
  Set environment variable: GEMINI_API_KEY=your_key_here

Fallback: mock mode for demos without internet or API key.

Usage (standalone):
    python reasoning.py                         # run demo query pair
    python reasoning.py --query "your question" # single query
    python reasoning.py --mock                  # mock mode (no API call)
"""

import re
import os
import argparse
from dataclasses import dataclass
from retrieval import tri_source_retrieve, format_context_for_llm

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a TSI compliance reasoning engine for EU railway certification.

You receive retrieved chunks from three independent document sources:

  - TSI:  LOC&PAS Commission Regulation (EU) 1302/2014 (consolidated to 2025) — English
  - NNTR: Arrêté du 19 mars 2012 fixant les objectifs, les méthodes, les indicateurs
          de sécurité et la réglementation technique applicable sur le réseau ferré national
          (French national rule) — French language
  - SPEC: IberRail IB-EMU-450 technical specification Rev. 4.0 — English
          (Cross-border authorisation submission for French RFN operation)

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

4. EN 14752: The European standard referenced in LOC&PAS TSI Art. 4.2.3.1 for door
   obstacle detection. Single worst-case position test only. Does NOT satisfy NF F31-054
   Section 6.3 for French RFN operation — this is Conflict 1.

RULES (non-negotiable):
- Cite the specific article/section number for EVERY factual claim
- If the same parameter is covered by both TSI and NNTR with different requirements,
  identify this as a CONFLICT and explain which takes precedence and why
- If a relevant article is missing from any collection, say so explicitly — do not invent
- Never guess. If evidence is insufficient, use RED tier and explain the gap
- Keep the reasoning concise — the NoBo assessor reads dozens of these per day
- If the NNTR chunk is in French, you may reason from it directly — do not refuse

TWO KNOWN CONFLICTS FOR THIS SCENARIO (IberRail IB-EMU-450, French RFN):

  CONFLICT 1 — Obstacle detection test protocol:
    TSI Art. 4.2.3.1 references EN 14752 Cl. 7.2 (single position, 150 N max, 30 mm).
    Arrêté 2012 Art. 49 via NF F31-054 Sec. 6.3 requires 5 height positions,
    100 N max per position, ≤1.5 J kinetic energy. IberRail FAT covers EN 14752 only.
    → Supplementary NF F31-054 testing required for French authorisation.

  CONFLICT 2 — Single-agent operation (CAS):
    TSI contains NO specific requirements for CAS door systems.
    Arrêté 2012 Art. 49 mandates NF F31-054 for ALL CAS-operated trains on RFN.
    French TER services are CAS-operated.
    → Four functional gaps must be resolved: platform surveillance, closure confirmation
      interlock, alarm door-lock behaviour, and re-closure dwell time.

OUTPUT FORMAT (strict — machine-parsed by confidence gating logic):

VERDICT: [COMPLIANT / CONFLICT DETECTED / INSUFFICIENT DATA]

EXPLANATION:
[2-4 sentences of reasoning with explicit article citations.
 If multiple conflicts exist, list each as a numbered item.
 Reference TSI, NNTR, and SPEC chunks by their collection label (e.g. [TSI-1], [NNTR-2]).]

RECOMMENDED ACTION:
[1-3 concrete actions the engineer must take. Be specific — name the standard, test,
 or document required. Avoid generic advice.]

CONFIDENCE: [GREEN / AMBER / RED] — [XX%] — [one-sentence reason for this tier]

CITATIONS: [comma-separated list of article/section references used]

CONFIDENCE TIER DEFINITIONS:
  GREEN  (>90%): All relevant articles found in retrieved context. Conflict position
                 clearly supported by source text. Safe for NoBo review.
  AMBER (70-90%): Relevant articles found but position involves inference across
                  sources, or one source is missing/incomplete. Requires elevated review.
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


# ── Mock response ─────────────────────────────────────────────────────────────

_MOCK_RESPONSE = """VERDICT: CONFLICT DETECTED

EXPLANATION:
1. Conflict 1 (Obstacle detection protocol): TSI Art. 4.2.3.1 references EN 14752 Cl. 7.2, which requires a single worst-case position test at ≤150 N and 30 mm minimum obstacle diameter [TSI-1]. Arrêté 19 mars 2012 Art. 49, via NF F31-054 Sec. 6.3, requires testing at 5 defined height positions (250, 500, 900, 1300, and 1600 mm above door sill) with a ≤100 N force limit and ≤1.5 J kinetic energy limit per position [NNTR-1]. IberRail's existing FAT (IB-EMU-450-TEST-012) satisfies EN 14752 only and does NOT satisfy NF F31-054 Sec. 6.3 [SPEC-1].
2. Conflict 2 (Single-agent operation): The LOC&PAS TSI contains no requirements specific to CAS-operated door systems. Art. 49 mandates NF F31-054 compliance for all CAS trains on the RFN, requiring platform surveillance via CCTV, two-step closure confirmation interlock, door lock-open on passenger alarm, and 5 s re-closure dwell [NNTR-2]. These are full gaps versus the TSI baseline.

RECOMMENDED ACTION:
1. Commission supplementary FAT at BODE GmbH per NF F31-054 Sec. 6.3 (5 height positions + kinetic energy measurement). Refer to open issue OI-003. Target: May 2026.
2. Confirm DCU-3100 software parameters CAS_CONFIRM_TIMEOUT ≤25 s and CAS_ALARM_LOCK = TRUE are set per CERTIFER's OI-004 findings before final NF F31-054 conformity assessment.
3. Do not submit AMEC application to EPSF until CERTIFER's final NF F31-054 assessment report is received.

CONFIDENCE: GREEN — 92% — Both conflicts are directly documented in retrieved NNTR Art. 49 and SPEC TS-004 chunks. No inference required.

CITATIONS: TSI Art. 4.2.3.1, Arrêté 2012 Art. 49, NF F31-054 Sec. 6.3, EN 14752 Cl. 7.2, IB-EMU-450-TS-004 Rev. 4.0 Sec. 4.2.3, IB-EMU-450-TS-004 Rev. 4.0 Sec. 4.3.1
"""


# ── Gemini API call ───────────────────────────────────────────────────────────

def _call_gemini(context: str, query: str) -> str:
    """
    Call Google Gemini 1.5 Flash via the free REST API.
    Requires GEMINI_API_KEY environment variable.
    Get a free key at: aistudio.google.com (no credit card needed).
    """
    import requests

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. "
            "Get a free key at aistudio.google.com and run: "
            "set GEMINI_API_KEY=your_key_here"
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"RETRIEVED CONTEXT:\n{context}\n\n"
                            f"COMPLIANCE QUERY:\n{query}"
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,   # low temp for consistent structured output
            "maxOutputTokens": 1000,
        }
    }

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── Public interface ──────────────────────────────────────────────────────────

def reason(
    query: str,
    n_chunks: int = 5,
    mock: bool = False,
) -> ComplianceResponse:
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
        Use for demos without internet or as backup if Gemini is unavailable.
    """
    retrieval = tri_source_retrieve(query, n=n_chunks)
    context   = format_context_for_llm(retrieval)

    if mock:
        raw = _MOCK_RESPONSE
    else:
        try:
            raw = _call_gemini(context, query)
        except Exception as e:
            # Graceful degradation: RED tier with error message
            raw = (
                f"VERDICT: INSUFFICIENT DATA\n\n"
                f"EXPLANATION:\nAPI call failed: {e}. Retrieved context is available "
                f"below for manual review.\n\n"
                f"RECOMMENDED ACTION:\nCheck GEMINI_API_KEY is set correctly and retry. "
                f"Run with --mock flag for demo without API.\n\n"
                f"CONFIDENCE: RED — 0% — API unavailable\n\n"
                f"CITATIONS: N/A"
            )

    return _parse_response(raw, query, context)


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
    parser.add_argument("--mock", "-m", action="store_true",
                        help="Use mock response (no API call)")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    demo_queries = [
        "Does IberRail's door obstacle detection testing satisfy French RFN requirements under Arrêté 2012 Article 49?",
        "What are the CAS single-agent operation requirements for the IB-EMU-450 on French TER services?",
    ]

    queries_to_run = [args.query] if args.query else demo_queries

    for q in queries_to_run:
        print(f"\nQuery: {q}\n")
        response = reason(q, n_chunks=args.n, mock=args.mock)
        print(format_response_display(response))
        allow, gate_msg = confidence_gate(response)
        print(f"\nGating: {gate_msg}\n")
