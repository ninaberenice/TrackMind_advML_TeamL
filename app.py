"""
app.py
======
TrackMind AI — Streamlit frontend.

Three-panel layout:
  LEFT    Document Context  — retrieved chunks from all three sources,
                              with language labels and article IDs
  CENTRE  Query + Response  — compliance query input, AI draft with
                              confidence tier badge, citations
  RIGHT   Assessor Review   — approve / reject / escalate + audit log

Run:
    streamlit run app.py

For demo mode (no API key, mock LLM):
    streamlit run app.py -- --mock
"""

import sys
import argparse
import streamlit as st
import chromadb

# Parse --mock flag before Streamlit captures argv
_MOCK = "--mock" in sys.argv or "-m" in sys.argv


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrackMind AI",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Tier badges */
  .badge-green  { background:#1a7a4a; color:white; padding:4px 12px;
                  border-radius:12px; font-weight:700; font-size:0.9em; }
  .badge-amber  { background:#c97a00; color:white; padding:4px 12px;
                  border-radius:12px; font-weight:700; font-size:0.9em; }
  .badge-red    { background:#c0392b; color:white; padding:4px 12px;
                  border-radius:12px; font-weight:700; font-size:0.9em; }

  /* Language labels */
  .lang-fr { background:#003189; color:white; padding:2px 8px;
             border-radius:6px; font-size:0.8em; }
  .lang-en { background:#012169; color:white; padding:2px 8px;
             border-radius:6px; font-size:0.8em; }

  /* Verdict colours */
  .verdict-conflict  { color:#c0392b; font-weight:700; font-size:1.1em; }
  .verdict-compliant { color:#1a7a4a; font-weight:700; font-size:1.1em; }
  .verdict-data      { color:#c97a00; font-weight:700; font-size:1.1em; }

  /* Source chunk cards */
  .chunk-card { background:#f8f9fa; border-left:3px solid #6c757d;
                padding:10px 14px; margin:6px 0; border-radius:4px;
                font-size:0.85em; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (avoid loading 2.5 GB model until needed) ───────────────────

@st.cache_resource(show_spinner="Loading bge-m3 multilingual model…")
def load_retrieval():
    from retrieval import tri_source_retrieve, format_context_for_llm, COLLECTIONS
    return tri_source_retrieve, format_context_for_llm, COLLECTIONS


@st.cache_resource(show_spinner=False)
def load_reasoning():
    from reasoning import reason, confidence_gate, format_response_display
    return reason, confidence_gate, format_response_display


@st.cache_resource(show_spinner=False)
def load_audit():
    from audit import log_full_interaction, get_recent_entries, get_stats
    return log_full_interaction, get_recent_entries, get_stats


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/100px-Flag_of_Europe.svg.png",
        width=40,
    )
    st.title("TrackMind AI")
    st.caption("TSI Compliance Intelligence")
    st.divider()

    if _MOCK:
        st.warning("🧪 **Mock mode** — LLM responses are canned. "
                   "Remove `--mock` for live API calls.")

    # Collection status
    st.subheader("Document Collections")
    try:
        chroma = chromadb.PersistentClient(path="./chroma_db")
        for name, label in [
            ("tsi_loc_pas",  "LOC&PAS TSI (EN)"),
            ("nntr_france",  "Arrêté 2012 (FR)"),
            ("spec_doc",     "IberRail Spec (EN)"),
        ]:
            col = chroma.get_or_create_collection(name)
            cnt = col.count()
            icon = "✅" if cnt > 0 else "⚠️"
            st.text(f"{icon} {label}: {cnt} chunks")
    except Exception as e:
        st.error(f"ChromaDB: {e}")

    st.divider()

    # Audit stats
    try:
        _, _, get_stats = load_audit()
        stats = get_stats()
        st.subheader("Audit Log")
        st.metric("Total queries",   stats["total_queries"])
        st.metric("Approvals",       stats["total_approvals"])
        st.metric("Escalations",     stats["total_escalations"])
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢", stats["green_count"])
        col2.metric("🟡", stats["amber_count"])
        col3.metric("🔴", stats["red_count"])
    except Exception:
        pass

    st.divider()

    # Pre-loaded demo queries
    st.subheader("Demo Queries")
    DEMO_QUERIES = [
        "Does IberRail's door obstacle detection testing satisfy French RFN requirements under Arrêté 2012 Article 49?",
        "What CAS single-agent operation requirements does the IB-EMU-450 need for French TER services?",
        "What Portuguese network national rules apply to IberRail's door system?",
    ]
    selected_demo = st.selectbox("Load a demo query:", [""] + DEMO_QUERIES, index=0)

    st.divider()
    n_chunks = st.slider("Chunks per collection", 3, 10, 5)
    assessor_id = st.text_input("Assessor ID", value="NoBo-Assessor-001")


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("🚂 TrackMind AI — Cross-Border Compliance Intelligence")
st.caption(
    "IberRail IB-EMU-450 · France RFN cross-border authorisation · "
    "Arrêté du 19 mars 2012 · LOC&PAS TSI (EU) 1302/2014"
)

left_col, centre_col, right_col = st.columns([3, 4, 3], gap="medium")


# ── Centre: Query input ───────────────────────────────────────────────────────
with centre_col:
    st.subheader("Compliance Query")

    # Pre-fill from sidebar demo selector
    default_query = selected_demo if selected_demo else ""
    query = st.text_area(
        "Enter compliance question (any language):",
        value=default_query,
        height=100,
        placeholder="e.g. Does IberRail's door obstacle detection satisfy French national requirements?",
    )

    run_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)

    if "response" not in st.session_state:
        st.session_state.response     = None
        st.session_state.retrieval    = None
        st.session_state.context_str  = None
        st.session_state.allow_draft  = False
        st.session_state.gate_message = ""
        st.session_state.logged_ids   = None


# ── Run retrieval + reasoning on button click ─────────────────────────────────
if run_btn and query.strip():
    tri_source_retrieve, format_context_for_llm, _ = load_retrieval()
    reason, confidence_gate, _ = load_reasoning()

    with st.spinner("Retrieving from 3 collections…"):
        retrieval = tri_source_retrieve(query, n=n_chunks)
        context   = format_context_for_llm(retrieval)

    with st.spinner("Reasoning across TSI + NNTR + Spec…"):
        response = reason(query, n_chunks=n_chunks, mock=_MOCK)

    allow, gate_msg = confidence_gate(response)

    st.session_state.response     = response
    st.session_state.retrieval    = retrieval
    st.session_state.context_str  = context
    st.session_state.allow_draft  = allow
    st.session_state.gate_message = gate_msg
    st.session_state.logged_ids   = None


# ── Centre: Response display ──────────────────────────────────────────────────
with centre_col:
    if st.session_state.response is not None:
        resp = st.session_state.response
        st.divider()

        # Verdict + confidence badge
        verdict_class = {
            "CONFLICT DETECTED": "verdict-conflict",
            "COMPLIANT":         "verdict-compliant",
            "INSUFFICIENT DATA": "verdict-data",
        }.get(resp.verdict, "verdict-data")

        tier_badge = {
            "GREEN": '<span class="badge-green">🟢 GREEN</span>',
            "AMBER": '<span class="badge-amber">🟡 AMBER</span>',
            "RED":   '<span class="badge-red">🔴 RED</span>',
        }.get(resp.confidence_tier, "")

        st.markdown(
            f'<p class="{verdict_class}">{resp.verdict}</p>'
            f'{tier_badge} &nbsp; <strong>{resp.confidence_pct}%</strong> confidence',
            unsafe_allow_html=True,
        )
        st.caption(f"Reason: {resp.confidence_reason}")

        # Gating message
        if not st.session_state.allow_draft:
            st.error(st.session_state.gate_message)
        else:
            if resp.confidence_tier == "AMBER":
                st.warning(st.session_state.gate_message)
            else:
                st.success(st.session_state.gate_message)

        st.divider()

        # Explanation
        st.subheader("Explanation")
        st.write(resp.explanation)

        # Recommended action
        st.subheader("Recommended Action")
        st.info(resp.recommended_action)

        # Citations
        if resp.citations:
            st.subheader("Citations")
            st.markdown(" · ".join(f"`{c}`" for c in resp.citations))

        # Raw response expander (for transparency)
        with st.expander("Raw LLM output (for GenAI Transparency Log)"):
            st.code(resp.raw_response, language="text")


# ── Left: Document context ────────────────────────────────────────────────────
with left_col:
    st.subheader("Document Context")

    if st.session_state.retrieval is not None:
        retrieval = st.session_state.retrieval

        for key, label, lang_code in [
            ("tsi",  "LOC&PAS TSI",         "en"),
            ("nntr", "Arrêté 19 mars 2012",  "fr"),
            ("spec", "IberRail IB-EMU-450",  "en"),
        ]:
            data = retrieval[key]
            lang_badge = (
                f'<span class="lang-fr">🇫🇷 French</span>'
                if lang_code == "fr"
                else f'<span class="lang-en">🇬🇧 English</span>'
            )

            with st.expander(f"{label}", expanded=(key == "nntr")):
                st.markdown(lang_badge, unsafe_allow_html=True)

                if data["empty"]:
                    st.warning("Collection empty — run ingestion first.")
                    continue

                for i, chunk in enumerate(data["chunks"], 1):
                    article_label = chunk["article"]
                    dist = chunk["distance"]
                    text_preview = chunk["text"][:400]

                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<strong>[{key.upper()}-{i}]</strong> '
                        f'<code>{article_label}</code> '
                        f'<small style="color:#888">dist={dist:.3f}</small><br>'
                        f'{text_preview}…'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Run a query to see retrieved document chunks here.")
        st.markdown("""
**What you'll see here:**

Three document layers retrieved in parallel:

- 🇬🇧 **LOC&PAS TSI** (English EU regulation)
- 🇫🇷 **Arrêté du 19 mars 2012** (French national rule — *cross-lingual retrieval*)
- 🇬🇧 **IberRail IB-EMU-450** (English technical spec)

The French text is retrieved by an English query — no translation layer.
This is the multilingual moat live on screen.
        """)


# ── Right: Assessor review panel ──────────────────────────────────────────────
with right_col:
    st.subheader("Assessor Review")

    if st.session_state.response is None:
        st.info("Submit a query to generate a compliance draft for review.")

    elif not st.session_state.allow_draft:
        st.error("🔴 RED tier — no draft to approve.")
        st.markdown("**Source documents returned.** Manual review required.")
        st.markdown(f"**Gap:** {st.session_state.response.confidence_reason}")
        st.divider()
        if st.button("📋 Log escalation to senior NoBo assessor",
                     use_container_width=True):
            log_full, _, _ = load_audit()
            resp = st.session_state.response
            retr = st.session_state.retrieval
            ids = log_full(
                query_text=resp.query,
                response=resp,
                assessor_id=assessor_id,
                decision="ESCALATED",
                notes="RED tier — insufficient evidence to draft",
                tsi_top=retr["tsi"]["chunks"][0]["article"] if retr["tsi"]["chunks"] else "",
                nntr_top=retr["nntr"]["chunks"][0]["article"] if retr["nntr"]["chunks"] else "",
                spec_top=retr["spec"]["chunks"][0]["article"] if retr["spec"]["chunks"] else "",
                mock=_MOCK,
            )
            st.session_state.logged_ids = ids
            st.success(f"Escalation logged (response_id={ids['response_id']})")

    else:
        # Show editable response
        st.caption("Review and edit the AI draft before approving.")
        edited_explanation = st.text_area(
            "Explanation (editable):",
            value=st.session_state.response.explanation,
            height=150,
        )
        edited_action = st.text_area(
            "Recommended Action (editable):",
            value=st.session_state.response.recommended_action,
            height=80,
        )
        assessor_notes = st.text_area(
            "Assessor notes (optional):",
            height=80,
            placeholder="e.g. Verified against NF F31-054 source document.",
        )

        col_a, col_r = st.columns(2)
        approve_btn = col_a.button(
            "✅ Approve", type="primary", use_container_width=True
        )
        reject_btn  = col_r.button(
            "❌ Reject", use_container_width=True
        )

        if approve_btn:
            log_full, _, _ = load_audit()
            resp = st.session_state.response
            retr = st.session_state.retrieval
            edited = f"EXPLANATION:\n{edited_explanation}\n\nRECOMMENDED ACTION:\n{edited_action}"
            ids = log_full(
                query_text=resp.query,
                response=resp,
                assessor_id=assessor_id,
                decision="APPROVED",
                notes=assessor_notes,
                edited_response=edited,
                tsi_top=retr["tsi"]["chunks"][0]["article"] if retr["tsi"]["chunks"] else "",
                nntr_top=retr["nntr"]["chunks"][0]["article"] if retr["nntr"]["chunks"] else "",
                spec_top=retr["spec"]["chunks"][0]["article"] if retr["spec"]["chunks"] else "",
                mock=_MOCK,
            )
            st.session_state.logged_ids = ids
            st.success(
                f"✅ **Approved** and logged to audit database.\n\n"
                f"Response ID: `{ids['response_id']}` · "
                f"Decision ID: `{ids['decision_id']}`"
            )

        if reject_btn:
            log_full, _, _ = load_audit()
            resp = st.session_state.response
            retr = st.session_state.retrieval
            ids = log_full(
                query_text=resp.query,
                response=resp,
                assessor_id=assessor_id,
                decision="REJECTED",
                notes=assessor_notes,
                tsi_top=retr["tsi"]["chunks"][0]["article"] if retr["tsi"]["chunks"] else "",
                nntr_top=retr["nntr"]["chunks"][0]["article"] if retr["nntr"]["chunks"] else "",
                spec_top=retr["spec"]["chunks"][0]["article"] if retr["spec"]["chunks"] else "",
                mock=_MOCK,
            )
            st.session_state.logged_ids = ids
            st.warning(
                f"❌ **Rejected** and logged.\n\n"
                f"Response ID: `{ids['response_id']}`"
            )

    # Recent audit log
    st.divider()
    st.subheader("Recent Audit Log")
    try:
        _, get_recent, _ = load_audit()
        entries = get_recent(5)
        if not entries:
            st.caption("No entries yet.")
        for e in entries:
            tier = e.get("confidence_tier") or "—"
            tier_icon = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(tier, "⚪")
            decision_icon = {
                "APPROVED": "✅", "REJECTED": "❌", "ESCALATED": "📋"
            }.get(e.get("decision"), "—")
            st.markdown(
                f"{tier_icon} `{str(e.get('query_text', ''))[:40]}…`  "
                f"{decision_icon} **{e.get('decision', '—')}**  "
                f"<small style='color:#888'>{str(e.get('query_ts', ''))[:16]}</small>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("Audit log unavailable.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "TrackMind AI · Nova School of Business and Economics · "
    "2758-T4 Advanced Topics in Machine Learning · May 2026  |  "
    "🚫 AI never auto-approves. Every compliance position requires NoBo assessor signature."
)
