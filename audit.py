"""
audit.py
========
SQLite audit log for TrackMind.

Every query, retrieved chunk set, AI draft, assessor decision, and final
response is stored here with timestamps. Satisfies the EU technical file
10-year retention requirement (Demo: SQLite; Production: PostgreSQL).

Schema
------
  queries      — one row per incoming query
  responses    — one row per AI-generated draft linked to a query
  decisions    — one row per assessor action (approve / reject)

Usage (standalone):
    python audit.py --show         # print last 10 audit entries
    python audit.py --export       # export to audit_log.json
    python audit.py --reset        # wipe the database (demo use only)
"""

import sqlite3
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path("./audit_log.db")


# ── Database initialisation ───────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Idempotent."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS queries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            query_text  TEXT NOT NULL,
            context_tsi_top  TEXT,
            context_nntr_top TEXT,
            context_spec_top TEXT
        );

        CREATE TABLE IF NOT EXISTS responses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            query_id        INTEGER NOT NULL REFERENCES queries(id),
            verdict         TEXT,
            explanation     TEXT,
            recommended_action TEXT,
            confidence_tier TEXT,
            confidence_pct  INTEGER,
            confidence_reason TEXT,
            citations       TEXT,
            raw_response    TEXT,
            mock            INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            response_id     INTEGER NOT NULL REFERENCES responses(id),
            assessor_id     TEXT NOT NULL,
            decision        TEXT NOT NULL,
            notes           TEXT,
            edited_response TEXT
        );
    """)
    conn.commit()
    conn.close()


# ── Write operations ──────────────────────────────────────────────────────────

def log_query(
    query_text: str,
    tsi_top_article: str = "",
    nntr_top_article: str = "",
    spec_top_article: str = "",
) -> int:
    """
    Log an incoming query. Returns the query row id for linking responses.
    """
    init_db()
    ts = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO queries
           (ts, query_text, context_tsi_top, context_nntr_top, context_spec_top)
           VALUES (?, ?, ?, ?, ?)""",
        (ts, query_text, tsi_top_article, nntr_top_article, spec_top_article),
    )
    query_id = cur.lastrowid
    conn.commit()
    conn.close()
    return query_id


def log_response(
    query_id: int,
    response,   # ComplianceResponse dataclass from reasoning.py
    mock: bool = False,
) -> int:
    """
    Log an AI-generated compliance response. Returns response row id.
    """
    init_db()
    ts = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO responses
           (ts, query_id, verdict, explanation, recommended_action,
            confidence_tier, confidence_pct, confidence_reason, citations,
            raw_response, mock)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ts,
            query_id,
            response.verdict,
            response.explanation,
            response.recommended_action,
            response.confidence_tier,
            response.confidence_pct,
            response.confidence_reason,
            json.dumps(response.citations),
            response.raw_response,
            int(mock),
        ),
    )
    response_id = cur.lastrowid
    conn.commit()
    conn.close()
    return response_id


def log_decision(
    response_id: int,
    assessor_id: str,
    decision: str,             # "APPROVED" / "REJECTED"
    notes: str = "",
    edited_response: str = "", # if assessor edited the draft before approving
) -> int:
    """
    Log an assessor decision. Returns decision row id.
    """
    if decision == "ESCALATED":
        decision = "REJECTED"
    init_db()
    ts = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO decisions
           (ts, response_id, assessor_id, decision, notes, edited_response)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (ts, response_id, assessor_id, decision, notes, edited_response),
    )
    decision_id = cur.lastrowid
    conn.commit()
    conn.close()
    return decision_id


# ── Convenience wrapper ───────────────────────────────────────────────────────

def log_full_interaction(
    query_text: str,
    response,
    assessor_id: str,
    decision: str,
    notes: str = "",
    edited_response: str = "",
    tsi_top: str = "",
    nntr_top: str = "",
    spec_top: str = "",
    mock: bool = False,
) -> dict:
    """
    One-call wrapper used by the Streamlit app: logs query → response → decision
    and returns all three IDs.
    """
    qid = log_query(query_text, tsi_top, nntr_top, spec_top)
    rid = log_response(qid, response, mock=mock)
    did = log_decision(rid, assessor_id, decision, notes, edited_response)
    return {"query_id": qid, "response_id": rid, "decision_id": did}


# ── Read operations ───────────────────────────────────────────────────────────

def get_recent_entries(n: int = 10) -> list[dict]:
    """Return the n most recent complete audit entries."""
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT
               q.ts            AS query_ts,
               q.query_text,
               r.verdict,
               r.confidence_tier,
               r.confidence_pct,
               r.citations,
               d.assessor_id,
               CASE WHEN d.decision='ESCALATED' THEN 'REJECTED' ELSE d.decision END AS decision,
               d.notes,
               d.ts            AS decision_ts
           FROM queries q
           LEFT JOIN responses r ON r.query_id = q.id
           LEFT JOIN decisions d ON d.response_id = r.id
           ORDER BY q.id DESC
           LIMIT ?""",
        (n,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def export_to_json(path: str = "audit_log.json") -> None:
    """Export full audit log to JSON for the business plan appendix."""
    init_db()
    conn = _get_conn()
    queries   = [dict(r) for r in conn.execute("SELECT * FROM queries").fetchall()]
    responses = [dict(r) for r in conn.execute("SELECT * FROM responses").fetchall()]
    decisions = [dict(r) for r in conn.execute("SELECT * FROM decisions").fetchall()]
    conn.close()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"queries": queries, "responses": responses, "decisions": decisions},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Exported {len(queries)} queries, {len(responses)} responses, "
          f"{len(decisions)} decisions to {path}")


def get_stats() -> dict:
    """Return summary statistics for display in the Streamlit sidebar."""
    init_db()
    conn = _get_conn()
    stats = {}
    stats["total_queries"]    = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    stats["total_responses"]  = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    stats["total_approvals"]  = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE decision='APPROVED'").fetchone()[0]
    stats["total_rejections"] = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE decision IN ('REJECTED', 'ESCALATED')").fetchone()[0]
    stats["green_count"] = conn.execute(
        "SELECT COUNT(*) FROM responses WHERE confidence_tier='GREEN'").fetchone()[0]
    stats["amber_count"] = conn.execute(
        "SELECT COUNT(*) FROM responses WHERE confidence_tier='AMBER'").fetchone()[0]
    stats["red_count"]   = conn.execute(
        "SELECT COUNT(*) FROM responses WHERE confidence_tier='RED'").fetchone()[0]
    conn.close()
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackMind audit log")
    parser.add_argument("--show",   action="store_true", help="Print last 10 entries")
    parser.add_argument("--export", action="store_true", help="Export to audit_log.json")
    parser.add_argument("--stats",  action="store_true", help="Show summary statistics")
    parser.add_argument("--reset",  action="store_true", help="Wipe database (demo only)")
    args = parser.parse_args()

    if args.reset:
        if DB_PATH.exists():
            DB_PATH.unlink()
            print("Audit database wiped.")
        else:
            print("No database to wipe.")

    elif args.export:
        export_to_json()

    elif args.stats:
        stats = get_stats()
        print("\n=== AUDIT LOG STATISTICS ===")
        for k, v in stats.items():
            print(f"  {k:25s}: {v}")

    else:  # --show or default
        entries = get_recent_entries(10)
        if not entries:
            print("Audit log is empty.")
        else:
            print(f"\n=== LAST {len(entries)} AUDIT ENTRIES ===\n")
            for e in entries:
                tier = e.get("confidence_tier") or "—"
                pct  = e.get("confidence_pct")  or "—"
                print(f"[{e['query_ts'][:19]}]")
                print(f"  Query:    {str(e['query_text'])[:80]}")
                print(f"  Verdict:  {e.get('verdict') or '—'}")
                print(f"  Conf:     {tier} {pct}%")
                print(f"  Assessor: {e.get('assessor_id') or '—'} → {e.get('decision') or '—'}")
                if e.get("notes"):
                    print(f"  Notes:    {e['notes'][:80]}")
                print()
