"""
test_chunking.py
================
Verification script for the TrackMind improved chunkers.

Run BEFORE re-running the full notebook to confirm that:
  1. france_nntr.pdf produces Art. 49 as a clean, standalone chunk
  2. Art. 49 contains the expected door-related content
  3. loc_pas_tsi.pdf produces section 4.2.5.5 and sub-sections as
     their own chunks (not merged blobs)
  4. No chunk in the NNTR collection has "section_N" as its article_id
     (the symptom of the original bug)
  5. No chunk starts with the page-header noise "28 mars 2012"

Usage:
    python test_chunking.py [--nntr path/to/france_nntr.pdf]
                            [--tsi  path/to/loc_pas_tsi.pdf]

Defaults assume the notebook's docs/ directory layout.
"""

import sys
import argparse
from trackmind_chunker import chunk_french_nntr, chunk_tsi_loc_pas

# ── Terminal colours (no external deps) ──────────────────────────────────────
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
RESET  = '\033[0m'
BOLD   = '\033[1m'


def ok(msg):   print(f'  {GREEN}✓ PASS{RESET}  {msg}')
def fail(msg): print(f'  {RED}✗ FAIL{RESET}  {msg}')
def warn(msg): print(f'  {YELLOW}⚠ WARN{RESET}  {msg}')
def section(msg): print(f'\n{BOLD}{msg}{RESET}')


# ── NNTR tests ────────────────────────────────────────────────────────────────

# Terms that MUST appear in Art. 49 (passenger door requirements).
# These come directly from the text of the Arrêté du 19 mars 2012 Art. 49(n).
ART49_REQUIRED_TERMS = [
    'portes',           # doors (sub-clause n: 'Les portes d'accès sont dotées...')
    'quais',            # platforms (sub-clause n: 'distance entre les quais et leurs portes')
    'emmarchement',     # step height (sub-clause n)
    'voyageurs',        # passengers (general throughout Art. 49)
    'sécurité',         # safety
]

# Terms that must NOT appear at the start of any chunk (page header noise).
FORBIDDEN_CHUNK_STARTS = [
    '28 mars 2012',
    'JOURNAL OFFICIEL',
    'Texte 36 sur 102',
]


def test_nntr(pdf_path: str) -> bool:
    section(f'═══ NNTR CHUNKING TESTS ({pdf_path}) ═══')
    passed = True

    chunks = chunk_french_nntr(pdf_path)

    # ── Test 1: Chunk count sanity ────────────────────────────────────────────
    if len(chunks) >= 100:
        ok(f'Chunk count {len(chunks)} ≥ 100 (expected ~120-130)')
    else:
        fail(f'Chunk count {len(chunks)} < 100 — likely over-merged; expected ~120-130')
        passed = False

    # ── Test 2: Art. 49 exists as exactly one chunk ───────────────────────────
    art49_chunks = [c for c in chunks if c['metadata']['article'] == 'Art. 49']
    if len(art49_chunks) == 1:
        ok(f'Art. 49 is exactly one chunk (id={art49_chunks[0]["id"]})')
    elif len(art49_chunks) == 0:
        fail('Art. 49 NOT FOUND as a chunk — split pattern did not fire on Art. 49')
        passed = False
    else:
        fail(f'Art. 49 found in {len(art49_chunks)} chunks (expected 1) — over-splitting')
        passed = False

    # ── Test 3: Art. 49 content verification ─────────────────────────────────
    if art49_chunks:
        art49_text = art49_chunks[0]['text'].lower()
        for term in ART49_REQUIRED_TERMS:
            if term in art49_text:
                ok(f'Art. 49 contains required term: "{term}"')
            else:
                fail(f'Art. 49 missing required term: "{term}"')
                passed = False

        # Check chunk starts with the actual Art. 49 header, not noise
        if art49_chunks[0]['text'].strip().startswith('Art. 49'):
            ok('Art. 49 chunk starts cleanly with "Art. 49"')
        else:
            fail(f'Art. 49 chunk does not start with "Art. 49": '
                 f'{repr(art49_chunks[0]["text"][:80])}')
            passed = False

        # Check Art. 49 chunk is not grotesquely large (should be ~4000-8000 chars)
        clen = len(art49_chunks[0]['text'])
        if 2000 <= clen <= 10000:
            ok(f'Art. 49 chunk length {clen} chars (reasonable: 2000–10000)')
        else:
            warn(f'Art. 49 chunk length {clen} chars is outside expected range 2000–10000')

    # ── Test 4: No section_N article IDs ────────────────────────────────────
    section_n = [c for c in chunks if c['metadata']['article'].startswith('section_')]
    if len(section_n) == 0:
        ok('No "section_N" article IDs — all chunks have real identifiers')
    elif len(section_n) <= 3:
        warn(f'{len(section_n)} chunk(s) have "section_N" id (acceptable for preamble/annexe)')
        # Show them
        for c in section_n:
            print(f'      {c["id"]}: {repr(c["text"][:60])}')
    else:
        fail(f'{len(section_n)} chunks have "section_N" id — '
             f'split pattern is missing many real articles')
        passed = False

    # ── Test 5: No chunk starts with page-header noise ───────────────────────
    noisy = [c for c in chunks
             if any(c['text'].strip().startswith(s) for s in FORBIDDEN_CHUNK_STARTS)]
    if len(noisy) == 0:
        ok('No chunks start with page-header noise')
    else:
        fail(f'{len(noisy)} chunk(s) start with page-header noise — '
             f'header stripping is incomplete')
        for c in noisy[:3]:
            print(f'      {c["id"]}: {repr(c["text"][:80])}')
        passed = False

    # ── Test 6: Key articles exist ────────────────────────────────────────────
    # NOTE: Art. 1 uses 'er' suffix in French: "Art. 1er". Test both forms.
    key_articles = [
        ('Art. 1', lambda ids: 'Art. 1er' in ids or 'Art. 1' in ids),
        ('Art. 9', lambda ids: 'Art. 9' in ids),
        ('Art. 47', lambda ids: 'Art. 47' in ids),
        ('Art. 48', lambda ids: 'Art. 48' in ids),
        ('Art. 49', lambda ids: 'Art. 49' in ids),
        ('Art. 50', lambda ids: 'Art. 50' in ids),
        ('Art. 100', lambda ids: 'Art. 100' in ids),
    ]
    found_ids = {c['metadata']['article'] for c in chunks}
    for art_label, check_fn in key_articles:
        if check_fn(found_ids):
            ok(f'{art_label} exists as a chunk')
        else:
            fail(f'{art_label} NOT found as a chunk')
            passed = False

    # ── Test 7: TITRE structural markers ────────────────────────────────────
    # TITRE Ier through TITRE VI — IDs are normalised to 'TITRE I' through 'TITRE VI'
    # because the regex stops at the first non-[IVX] char ('e' in 'Ier').
    # IDs are normalised: 'TITRE Ier' → 'TITRE I', 'TITRE II' → 'TITRE II', etc.
    import re as _re
    titres = [c for c in chunks
              if _re.match(r'TITRE\s+[IVX]', c['metadata']['article'], _re.IGNORECASE)]
    if len(titres) >= 6:
        ok(f'{len(titres)} TITRE markers found as chunks (all 6 present)')
    elif len(titres) >= 4:
        warn(f'{len(titres)} TITRE markers found (expected 6, got partial)')
    else:
        fail(f'Only {len(titres)} TITRE markers found (expected 6)')

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    if passed:
        print(f'  {GREEN}{BOLD}NNTR: ALL TESTS PASSED{RESET}')
    else:
        print(f'  {RED}{BOLD}NNTR: SOME TESTS FAILED — do not re-run notebook yet{RESET}')

    return passed


# ── TSI tests ─────────────────────────────────────────────────────────────────

TSI_KEY_SECTIONS = [
    '4.2.5.5',      # Exterior doors (top-level)
    '4.2.5.5.1',    # General
    '4.2.5.5.3',    # Door closing and locking
    '4.2.5.5.6',    # Door opening
    '4.2.5.5.9',    # Door emergency opening
]

# Terms that should appear somewhere in the 4.2.5.5 cluster
TSI_DOOR_REQUIRED_TERMS = ['door', 'obstacle', 'passenger', 'closing']


def test_tsi(pdf_path: str) -> bool:
    section(f'═══ TSI CHUNKING TESTS ({pdf_path}) ═══')
    passed = True

    chunks = chunk_tsi_loc_pas(pdf_path)

    # ── Test 1: Chunk count sanity ────────────────────────────────────────────
    if 800 <= len(chunks) <= 1500:
        ok(f'Chunk count {len(chunks)} in expected range 800–1500')
    else:
        warn(f'Chunk count {len(chunks)} outside expected range 800–1500')

    # ── Test 2: Key door sections exist ──────────────────────────────────────
    found_ids = {c['metadata']['article'] for c in chunks}
    for sec in TSI_KEY_SECTIONS:
        # Match exact or with trailing digits (4.2.5.5 should match 4.2.5.5.x chunks)
        matches = [c for c in chunks
                   if c['metadata']['article'] == sec
                   or c['metadata']['article'].startswith(sec + '.')]
        if matches:
            ok(f'Section {sec} found ({len(matches)} chunk(s))')
        else:
            fail(f'Section {sec} NOT found — door content may be missing')
            passed = False

    # ── Test 3: Obstacle detection content ───────────────────────────────────
    obs_chunks = [c for c in chunks if 'obstacle' in c['text'].lower()]
    if len(obs_chunks) >= 1:
        ok(f'Obstacle detection content found in {len(obs_chunks)} chunk(s)')
        # Check the best candidate
        best = obs_chunks[0]
        print(f'    Best chunk: {best["metadata"]["article"]} — '
              f'{repr(best["text"][:100])}')
    else:
        fail('No chunk contains "obstacle" — door obstacle detection content missing')
        passed = False

    # ── Test 4: No EUR-Lex amendment markers in chunk text ───────────────────
    import re
    marker_re = re.compile(r'[▼►◄]\s*[MB]\d*')
    noisy = [c for c in chunks if marker_re.search(c['text'])]
    if len(noisy) == 0:
        ok('No EUR-Lex amendment markers in chunk text')
    elif len(noisy) <= 10:
        warn(f'{len(noisy)} chunk(s) contain EUR-Lex markers (minor noise, acceptable)')
    else:
        fail(f'{len(noisy)} chunk(s) contain EUR-Lex amendment markers '
             f'— stripping is incomplete')
        passed = False

    # ── Test 5: No version-stamp starts ──────────────────────────────────────
    stamp_chunks = [c for c in chunks if c['text'].startswith('02014R1302')]
    if len(stamp_chunks) == 0:
        ok('No chunks start with EUR-Lex version stamp')
    else:
        fail(f'{len(stamp_chunks)} chunk(s) start with version stamp '
             f'"02014R1302…" — noise not stripped')
        passed = False

    print()
    if passed:
        print(f'  {GREEN}{BOLD}TSI: ALL TESTS PASSED{RESET}')
    else:
        print(f'  {RED}{BOLD}TSI: SOME TESTS FAILED{RESET}')

    return passed


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Verify TrackMind chunking before notebook re-run'
    )
    parser.add_argument('--nntr', default='docs/france_nntr.pdf',
                        help='Path to france_nntr.pdf')
    parser.add_argument('--tsi', default='docs/loc_pas_tsi.pdf',
                        help='Path to loc_pas_tsi.pdf')
    parser.add_argument('--skip-tsi', action='store_true',
                        help='Skip TSI tests (only test NNTR)')
    args = parser.parse_args()

    import os
    all_passed = True

    if os.path.exists(args.nntr):
        all_passed &= test_nntr(args.nntr)
    else:
        print(f'\n{RED}NNTR file not found: {args.nntr}{RESET}')
        all_passed = False

    if not args.skip_tsi:
        if os.path.exists(args.tsi):
            all_passed &= test_tsi(args.tsi)
        else:
            print(f'\n{YELLOW}TSI file not found: {args.tsi} — skipping TSI tests{RESET}')

    print('\n' + '═' * 60)
    if all_passed:
        print(f'{GREEN}{BOLD}ALL TESTS PASSED — safe to re-run notebook{RESET}')
        sys.exit(0)
    else:
        print(f'{RED}{BOLD}TESTS FAILED — fix chunker before re-running notebook{RESET}')
        sys.exit(1)


if __name__ == '__main__':
    main()
