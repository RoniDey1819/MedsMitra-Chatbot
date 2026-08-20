"""
Crawls the pharmacy's own website (auto-discovering internal links from a
seed URL), extracts readable text, chunks it, embeds it locally, and
upserts it into the Supabase `website_content` table (see
supabase_setup.sql for the schema).

This is what lets the chatbot answer general questions — store hours,
location, services, return policy, etc. — using content that gets merged
with medicine search results in app.py's retrieve_context().

Standalone run (one-time or manual refresh):
    python crawl_site.py

Scheduled run: app.py can call run_crawl() on a daily cron schedule via
APScheduler when ENABLE_WEBSITE_CRAWL_SCHEDULER=true (see app.py). In that
case it reuses the already-loaded embedding model instead of loading a
second copy of it.
"""

import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import psycopg2
import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("crawl_site")

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # same Supabase connection string as load_data.py
WEBSITE_URL = os.getenv("WEBSITE_URL")  # seed URL, e.g. https://mypharmacy.com

CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "200"))
CRAWL_MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "3"))
CRAWL_DELAY_SECONDS = float(os.getenv("CRAWL_DELAY_SECONDS", "0.5"))  # politeness delay between requests
CRAWL_TIMEOUT_SECONDS = float(os.getenv("CRAWL_TIMEOUT_SECONDS", "10"))
CRAWL_USER_AGENT = os.getenv(
    "CRAWL_USER_AGENT", "MedsMitraBot/1.0 (+website content indexer for pharmacy chatbot)"
)

CHUNK_SIZE = int(os.getenv("CRAWL_CHUNK_SIZE", "800"))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CRAWL_CHUNK_OVERLAP", "100"))  # characters of overlap between chunks
MIN_CHUNK_LENGTH = int(os.getenv("CRAWL_MIN_CHUNK_LENGTH", "40"))

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
DB_PAGE_SIZE = int(os.getenv("DB_PAGE_SIZE", "100"))

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Extensions that are never worth fetching as "pages" (files, media, etc.)
_SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".rar", ".doc", ".docx", ".xls", ".xlsx", ".mp4", ".mp3",
    ".css", ".js", ".woff", ".woff2", ".ttf",
)
_SKIP_SCHEMES = ("mailto:", "tel:", "javascript:", "#")


# -------------------------------------------------------------------
# Fetching & parsing
# -------------------------------------------------------------------


def _normalize(url: str) -> str:
    """Strip fragment and trailing slash so equivalent URLs dedupe."""
    url, _ = urldefrag(url)
    if url.endswith("/") and url.count("/") > 2:
        url = url[:-1]
    return url


def _same_domain(url: str, domain: str) -> bool:
    return urlparse(url).netloc == domain


def _is_crawlable(url: str) -> bool:
    if any(url.lower().startswith(s) for s in _SKIP_SCHEMES):
        return False
    path = urlparse(url).path.lower()
    return not path.endswith(_SKIP_EXTENSIONS)


def _fetch(session: requests.Session, url: str) -> Optional[str]:
    try:
        resp = session.get(url, timeout=CRAWL_TIMEOUT_SECONDS)
    except requests.RequestException:
        logger.warning("Fetch failed: %s", url)
        return None

    content_type = resp.headers.get("content-type", "")
    if resp.status_code != 200 or "text/html" not in content_type:
        return None
    return resp.text


def _extract_lab_tests(soup: BeautifulSoup) -> list[str]:
    """Pulls structured lab-test data out of plain `.tile` blocks on the lab
    tests page. Unlike product/doctor cards, these have no dedicated class
    (just `.tile` with inline styles), and the page mixes two shapes:

      - Health packages: `.badge-soft` (parameter count / audience tag) +
        `h5` name + description `<p>` + price/MRP.
      - Individual tests: `h6` name + price/MRP only, no badge or
        description.

    We key off `.badge-soft` to tell the two apart, and require a price
    element to exist at all (this excludes the `.filter-card` order-summary
    sidebar, which is a different tile-like block with no price/name to
    extract and would otherwise pollute the chunk set).
    """
    chunks = []
    for tile in soup.select(".tile"):
        price_el = tile.select_one(".price")
        if not price_el:
            continue  # not a purchasable test/package tile (e.g. order-summary sidebar)

        mrp_el = tile.select_one(".mrp")
        price = price_el.get_text(strip=True)
        mrp = mrp_el.get_text(strip=True) if mrp_el else ""

        badge_el = tile.select_one(".badge-soft")
        if badge_el:
            # Health package: has a name in h5 and a description paragraph.
            name_el = tile.select_one("h5")
            if not name_el:
                continue
            name = name_el.get_text(strip=True)
            badge = badge_el.get_text(strip=True)
            desc_el = tile.select_one("p")
            description = desc_el.get_text(strip=True) if desc_el else ""

            sentence = f"Health Package: {name} ({badge})."
            if description:
                sentence += f" Includes: {description}"
            sentence += f" Price: {price}"
        else:
            # Individual test: name is in h6, no description.
            name_el = tile.select_one("h6")
            if not name_el:
                continue
            name = name_el.get_text(strip=True)
            sentence = f"Lab Test: {name}. Price: {price}"

        if mrp and mrp != price:
            sentence += f" (MRP {mrp})"
        sentence += ". Free home sample collection, reports in 24 hours."

        chunks.append(sentence)

    return chunks


def _extract_doctors(soup: BeautifulSoup) -> list[str]:
    """Pulls structured doctor data out of `.tile.doctor-card` blocks (name /
    specialty / rating / experience) and turns each into one clean,
    self-contained sentence — same rationale as _extract_products(): a page
    listing several doctors becomes one diluted paragraph without this, so a
    query like "name a general physician" has to match a blob describing
    every specialty at once instead of one focused sentence about Dr. Anita
    Sharma specifically. One doctor = one chunk fixes that.
    """
    chunks = []
    for card in soup.select(".tile.doctor-card"):
        name_el = card.select_one("h6")
        if not name_el:
            continue
        name = name_el.get_text(strip=True)

        meta_el = card.select_one(".meta")
        specialty = meta_el.get_text(strip=True) if meta_el else ""

        rating_el = card.select_one(".rating")
        rating = rating_el.get_text(strip=True) if rating_el else ""

        exp_el = card.select_one(".exp")
        experience = exp_el.get_text(strip=True) if exp_el else ""

        sentence = f"Doctor: {name}."
        if specialty:
            sentence += f" Specialty: {specialty}."
        if experience:
            sentence += f" {experience}."
        if rating:
            sentence += f" Rating: {rating}."
        sentence += " Available for chat, audio, or video consultation."

        chunks.append(sentence)

    return chunks


def _extract_products(soup: BeautifulSoup) -> list[str]:
    """Pulls structured product data out of `.tile.product-card` blocks
    (name / composition-or-category / brand / price / MRP / discount /
    Rx-required) and turns each into one clean, self-contained sentence.

    This matters because without it, a page listing 12+ products becomes a
    single diluted paragraph mixing every product's name and price
    together — a query like "price of azithral" then has to match against
    a blob about 12 unrelated drugs instead of one focused sentence about
    Azithral, which hurts retrieval and invites the model to mix up prices
    between products. One product = one chunk fixes both problems.
    """
    chunks = []
    for card in soup.select(".tile.product-card"):
        name_el = card.select_one(".name")
        if not name_el:
            continue
        name = name_el.get_text(strip=True)

        meta_el = card.select_one(".meta")
        meta = meta_el.get_text(strip=True) if meta_el else ""

        price_el = card.select_one(".price")
        mrp_el = card.select_one(".mrp")
        discount_el = card.select_one(".rx-tag")
        price = price_el.get_text(strip=True) if price_el else ""
        mrp = mrp_el.get_text(strip=True) if mrp_el else ""
        discount = discount_el.get_text(strip=True) if discount_el else ""

        rx_required = "rx required" in meta.lower()

        sentence = f"Product: {name}."
        if meta:
            sentence += f" {meta}."
        if price:
            sentence += f" Price: {price}"
            if mrp and mrp != price:
                sentence += f" (MRP {mrp}"
                if discount:
                    sentence += f", {discount}"
                sentence += ")"
            sentence += "."
        sentence += " Prescription required." if rx_required else " No prescription required."

        chunks.append(sentence)

    return chunks


def _extract_content(html: str, url: str) -> tuple[str, list[str]]:
    """Returns (title, chunks) for a page: structured per-product,
    per-doctor, and per-lab-test/package chunks (if any such cards are
    present) plus chunked general page text (intro copy, FAQs, policies,
    etc.) extracted via trafilatura from the remaining HTML with those
    cards stripped out, so the general text doesn't just re-duplicate a
    diluted version of the same product/doctor/test list.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    product_chunks = _extract_products(soup)
    doctor_chunks = _extract_doctors(soup)
    lab_test_chunks = _extract_lab_tests(soup)

    for card in soup.select(".tile.product-card"):
        card.decompose()
    for card in soup.select(".tile.doctor-card"):
        card.decompose()
    # Lab test tiles are plain `.tile` with no dedicated class, so only
    # decompose the ones we actually extracted from (has a `.price`) —
    # this leaves the `.filter-card` sidebar and any other `.tile` blocks
    # without prices untouched for the general-text fallback below.
    for tile in soup.select(".tile"):
        if tile.select_one(".price"):
            tile.decompose()

    remaining_html = str(soup)
    general_text = trafilatura.extract(
        remaining_html, url=url, include_comments=False, include_tables=False
    ) or ""

    if not general_text.strip():
        strip_soup = BeautifulSoup(remaining_html, "html.parser")
        for tag in strip_soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        general_text = strip_soup.get_text(separator="\n", strip=True)

    general_chunks = _chunk_text(general_text)

    return title, product_chunks + doctor_chunks + lab_test_chunks + general_chunks


def _extract_links(html: str, base_url: str, domain: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(base_url, a["href"])
        if _same_domain(absolute, domain) and _is_crawlable(absolute):
            links.append(absolute)
    return links


def _crawl(seed_url: str) -> dict[str, tuple[str, list[str]]]:
    """Breadth-first crawl of the site starting at seed_url, staying on the
    same domain and respecting robots.txt. Returns {url: (title, chunks)}."""
    domain = urlparse(seed_url).netloc

    robots = RobotFileParser()
    robots.set_url(urljoin(seed_url, "/robots.txt"))
    try:
        robots.read()
    except Exception:
        logger.warning("Could not read robots.txt for %s — proceeding without it.", domain)

    session = requests.Session()
    session.headers.update({"User-Agent": CRAWL_USER_AGENT})

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(seed_url, 0)])
    pages: dict[str, tuple[str, list[str]]] = {}

    while queue and len(pages) < CRAWL_MAX_PAGES:
        url, depth = queue.popleft()
        norm = _normalize(url)
        if norm in visited or depth > CRAWL_MAX_DEPTH:
            continue
        visited.add(norm)

        try:
            if not robots.can_fetch(CRAWL_USER_AGENT, url):
                logger.info("Skipping (robots.txt disallows): %s", url)
                continue
        except Exception:
            pass

        html = _fetch(session, url)
        time.sleep(CRAWL_DELAY_SECONDS)
        if not html:
            continue

        title, chunks = _extract_content(html, url)
        if chunks:
            pages[norm] = (title, chunks)
            logger.info(
                "Crawled (%d/%d): %s (%d chunk(s))", len(pages), CRAWL_MAX_PAGES, url, len(chunks)
            )

        if depth < CRAWL_MAX_DEPTH:
            for link in _extract_links(html, url, domain):
                if _normalize(link) not in visited:
                    queue.append((link, depth + 1))

    return pages


# -------------------------------------------------------------------
# Chunking
# -------------------------------------------------------------------


def _chunk_text(text: str) -> list[str]:
    """Splits text into overlapping character-based chunks, breaking on a
    word boundary near the target size so words aren't cut mid-way."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) <= CHUNK_SIZE:
        return [text] if len(text) >= MIN_CHUNK_LENGTH else []

    chunks = []
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
        start += step

    return chunks


# -------------------------------------------------------------------
# Supabase upsert
# -------------------------------------------------------------------


def _upsert_records(
    records: list[tuple[str, str, int, str]],  # (url, title, chunk_index, content)
    embeddings: list[list[float]],
    crawled_urls: list[str],
    crawl_start: datetime,
) -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()

        # Clear old chunks for every URL touched this run, so a page that
        # shrank (fewer chunks than before) doesn't leave stale rows behind.
        cur.execute("delete from website_content where url = any(%s)", (crawled_urls,))

        insert_sql = """
            insert into website_content (url, title, chunk_index, content, embedding, crawled_at)
            values %s
        """
        values = [
            (url, title, idx, content, emb, crawl_start)
            for (url, title, idx, content), emb in zip(records, embeddings)
        ]
        execute_values(cur, insert_sql, values, page_size=DB_PAGE_SIZE)

        # Anything not touched by this run (page removed, moved, or no
        # longer reachable) is now stale — clean it up.
        cur.execute("delete from website_content where crawled_at < %s", (crawl_start,))
        deleted = cur.rowcount
        if deleted:
            logger.info("Removed %d stale chunk(s) from pages no longer found.", deleted)

        conn.commit()
        cur.close()
    finally:
        conn.close()


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------


def run_crawl(embed_model: Optional[SentenceTransformer] = None) -> None:
    """Crawls WEBSITE_URL and refreshes the website_content table.

    Pass an already-loaded embed_model (e.g. from app.py) to avoid loading
    a second copy of sentence-transformers when called from a scheduler.
    """
    if not WEBSITE_URL:
        logger.warning("WEBSITE_URL is not set — skipping website crawl.")
        return
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set. Add your Supabase connection string to .env")

    crawl_start = datetime.now(timezone.utc)
    logger.info(
        "Starting crawl of %s (max_pages=%d, max_depth=%d)",
        WEBSITE_URL, CRAWL_MAX_PAGES, CRAWL_MAX_DEPTH,
    )

    pages = _crawl(WEBSITE_URL)
    if not pages:
        logger.warning("Crawl found no pages — nothing to update.")
        return

    records: list[tuple[str, str, int, str]] = []
    for url, (title, chunks) in pages.items():
        for i, chunk in enumerate(chunks):
            records.append((url, title, i, chunk))

    if not records:
        logger.warning("No usable text extracted from any crawled page.")
        return

    logger.info("Extracted %d chunks from %d pages.", len(records), len(pages))

    own_model = embed_model is None
    if own_model:
        logger.info("Loading embedding model...")
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [r[3] for r in records]
    logger.info("Embedding %d chunks in batches of %d...", len(texts), EMBED_BATCH_SIZE)
    raw_embeddings = embed_model.encode(
        texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True,
    )
    embeddings = [e.tolist() for e in raw_embeddings]

    _upsert_records(records, embeddings, list(pages.keys()), crawl_start)
    logger.info("Website crawl complete — %d chunks from %d pages upserted.", len(records), len(pages))


if __name__ == "__main__":
    run_crawl()