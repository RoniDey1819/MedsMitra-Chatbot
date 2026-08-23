"""
Fuzzy spell-correction for medicine names typed by users.

Why not a generic English spellchecker: "Paracetamol" and "Azithral" aren't
in a standard dictionary, so a generic checker either flags real medicine
names as errors or can't correct genuine typos of them. Instead we fuzzy-
match user input directly against the pharmacy's own medicine name list
pulled from Supabase, so corrections are always grounded in what's actually
in stock/catalogued.

Handles two distinct typo patterns:
1. Spaced-out typos: "is percitamul avialble?" -> individual words are
   fuzzy-matched against known medicine names.
2. Mashed-together strings: "isparacetamolavialble" -> no word boundaries
   exist, so instead we slide a window of varying length across the string
   and fuzzy-match each window against known medicine names.

Usage:
    corrector = SpellCorrector(supabase_client)
    corrector.refresh()  # call once at startup, then periodically

    result = corrector.correct("is percitamul avialble?")
    # result.corrected_text -> "is paracetamol avialble?"
    # result.matched_name   -> "Paracetamol"
    # result.confidence     -> 0.87
    # result.should_confirm -> True   (ambiguous enough to ask "did you mean?")
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

logger = logging.getLogger("medsmitra.spell_correct")

# Below this score, we don't think it's a match at all — leave text alone.
MIN_MATCH_SCORE = 62
# Above this score, treat it as basically certain (typo, not a different
# medicine) — correct silently without asking for confirmation.
AUTO_CORRECT_SCORE = 90
# At or above MIN_MATCH_SCORE but below AUTO_CORRECT_SCORE: correct AND
# flag should_confirm=True so the caller can append a "did you mean?" note.

# Plain edit-distance (fuzz.ratio) can't reliably tell apart phonetically
# different drug names — e.g. "azitromycin" scores HIGHER against "Crocin"
# than against the phonetically obvious "Azithral" using ratio alone, since
# both share several letters in any order. Typos in drug names typically
# preserve the start of the word even when the middle/end drifts (a user
# who mis-hears or mis-types "Azithromycin" still starts typing "azi..."),
# so we weight the first few characters more heavily than the full-string
# ratio. This is a heuristic, not phonetic matching, but it resolves the
# specific failure mode observed above without adding a phonetic-algorithm
# dependency.
_PREFIX_LEN = 3
_PREFIX_WEIGHT = 0.6   # weight given to the prefix-only comparison
_BASE_WEIGHT = 1 - _PREFIX_WEIGHT


def _match_score(candidate: str, name: str) -> float:
    candidate, name = candidate.lower(), name.lower()
    base = fuzz.ratio(candidate, name)
    prefix_score = fuzz.ratio(candidate[:_PREFIX_LEN], name[:_PREFIX_LEN])
    return _BASE_WEIGHT * base + _PREFIX_WEIGHT * prefix_score


def _best_match(candidate: str, names: list[str]) -> Optional[tuple[str, float]]:
    """Returns (best_name, score) or None if names is empty."""
    best = None
    for name in names:
        score = _match_score(candidate, name)
        if best is None or score > best[1]:
            best = (name, score)
    return best


_WORD_PATTERN = re.compile(r"[A-Za-z]+")

# Matches the exact "Product: <name>." prefix that crawl_site.py's
# _extract_products() generates for each product card — see that file's
# sentence-building logic. This is how brand names that only exist on the
# live website (not in medicines.csv) still make it into the correction
# cache, e.g. "Crocin Advance" or "Zincovit" as sold/priced on-site even
# if medicines.csv only lists the generic "Paracetamol"/"Vitamin C".
_PRODUCT_NAME_PATTERN = re.compile(r"Product:\s*([A-Za-z0-9][\w\- ]*?)\.")


@dataclass
class CorrectionResult:
    corrected_text: str
    matched_name: Optional[str] = None
    confidence: float = 0.0
    should_confirm: bool = False
    original_text: str = ""

    @property
    def changed(self) -> bool:
        return self.corrected_text != self.original_text


class SpellCorrector:
    def __init__(self, supabase_client, refresh_interval_seconds: int = 3600):
        self._supabase = supabase_client
        self._refresh_interval = refresh_interval_seconds
        self._names: list[str] = []
        self._lock = threading.Lock()
        self._last_refresh = 0.0

    def refresh(self) -> None:
        """Pulls known medicine/brand names from Supabase. Combines several
        sources because a single 'medicine_name' column is not enough:
        users search by brand name ("Crocin", "Zincovit") which often only
        appears in the medicines table's 'alternative' column (comma-
        separated brand/alternative names — see load_data.py's schema), or
        only in website_content (crawled product listing pages) if it's not
        in the medicines table at all. Safe to call repeatedly; failures are
        logged and the previous cache is kept."""
        names: set[str] = set()

        try:
            resp = (
                self._supabase.table("medicines")
                .select("medicine_name, alternative")
                .execute()
            )
            for row in resp.data or []:
                for col in ("medicine_name", "alternative"):
                    raw = (row.get(col) or "").strip()
                    if not raw:
                        continue
                    # 'alternative' can be a comma-separated list, e.g.
                    # "Crocin, Dolo" — split it into individual names.
                    for part in raw.split(","):
                        part = part.strip()
                        if part:
                            names.add(part)
        except Exception:
            logger.exception("Spell-correction refresh: medicines table query failed.")

        try:
            resp = (
                self._supabase.table("website_content")
                .select("content")
                .execute()
            )
            for row in resp.data or []:
                content = row.get("content") or ""
                match = _PRODUCT_NAME_PATTERN.search(content)
                if match:
                    names.add(match.group(1).strip())
        except Exception:
            # website_content may not exist yet, or the crawl hasn't run —
            # this source is optional, so don't let it block the medicines
            # table results above.
            logger.warning("Spell-correction refresh: website_content query skipped/failed.", exc_info=True)

        if names:
            with self._lock:
                self._names = sorted(names)
                self._last_refresh = time.time()
            logger.info("Spell-correction cache refreshed: %d known names (medicines + website).", len(names))
        else:
            logger.warning("Spell-correction refresh returned no names at all — keeping old cache.")

    def _maybe_refresh(self) -> None:
        if time.time() - self._last_refresh > self._refresh_interval:
            self.refresh()

    def _known_names(self) -> list[str]:
        with self._lock:
            return list(self._names)

    def correct(self, text: str) -> CorrectionResult:
        """Best-effort fuzzy correction. Never raises — on any failure or
        empty cache, returns the text unchanged."""
        self._maybe_refresh()
        names = self._known_names()
        if not names:
            return CorrectionResult(corrected_text=text, original_text=text)

        try:
            # --- Pass 1: word-level fuzzy match (handles "percitamul") ---
            words = _WORD_PATTERN.findall(text)
            best_overall: Optional[tuple[str, str, float]] = None  # (raw_word, matched_name, score)

            for word in words:
                if len(word) < 4:  # too short to meaningfully fuzzy-match
                    continue
                match = _best_match(word, names)
                if not match:
                    continue
                matched_name, score = match
                if word.lower() == matched_name.lower():
                    continue  # already correct, nothing to do
                if score >= MIN_MATCH_SCORE and (best_overall is None or score > best_overall[2]):
                    best_overall = (word, matched_name, score)

            if best_overall:
                raw_word, matched_name, score = best_overall
                corrected_text = re.sub(
                    re.escape(raw_word), matched_name, text, count=1, flags=re.I
                )
                return CorrectionResult(
                    corrected_text=corrected_text,
                    matched_name=matched_name,
                    confidence=score / 100,
                    should_confirm=score < AUTO_CORRECT_SCORE,
                    original_text=text,
                )

            # --- Pass 2: mashed-together string (handles "isparacetamolavialble") ---
            compact = "".join(words).lower()
            if len(words) <= 2 and len(compact) >= 12:
                best_window: Optional[tuple[str, float]] = None
                for name in names:
                    name_lower = name.lower().replace(" ", "")
                    window_len = len(name_lower)
                    if window_len < 4:
                        continue
                    # Slide a window roughly the length of this medicine
                    # name across the compact string looking for a fuzzy hit.
                    for start in range(0, max(len(compact) - window_len + 1, 1)):
                        window = compact[start:start + window_len]
                        score = _match_score(window, name_lower)
                        if score >= MIN_MATCH_SCORE and (best_window is None or score > best_window[1]):
                            best_window = (name, score)
                if best_window:
                    matched_name, score = best_window
                    # We can't cleanly splice a correction into a mashed
                    # string, so we replace the whole thing with a spaced-out
                    # best-guess: the matched medicine name plus a generic
                    # availability phrasing. This keeps retrieval working;
                    # the "did you mean?" note tells the user what happened.
                    corrected_text = f"is {matched_name} available?"
                    return CorrectionResult(
                        corrected_text=corrected_text,
                        matched_name=matched_name,
                        confidence=score / 100,
                        should_confirm=True,  # always confirm — this is a lossy guess
                        original_text=text,
                    )

        except Exception:
            logger.exception("Spell correction failed for text=%r — returning unchanged.", text)

        return CorrectionResult(corrected_text=text, original_text=text)