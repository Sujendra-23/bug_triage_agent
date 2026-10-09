"""
crawler.py — Sitemap-driven crawler that keeps a ChromaDB collection in sync with a docs site.

Flow: sitemap(s) -> page URLs -> fetch -> clean HTML -> chunk -> embed into ChromaDB.
Re-running is cheap and safe:
  * a page whose cleaned text hash is unchanged is skipped (no re-embedding)
  * a changed page has its old chunks replaced
  * a page that left the sitemap, or now returns 404/410, has its chunks removed
  * if the sitemap itself cannot be fetched, nothing is deleted

Only same-host http(s) URLs are fetched, robots.txt is honoured, and requests are spaced out.
"""

import hashlib
import re
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup

USER_AGENT = "bug-triage-agent-crawler/1.0"
MAX_SITEMAP_DEPTH = 3

# fetch(url) -> (status_code, body_text). Injectable so tests never touch the network.
Fetcher = Callable[[str], Tuple[int, str]]


def default_fetcher(timeout: float = 15.0) -> Fetcher:
    import requests

    def fetch(url: str) -> Tuple[int, str]:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        return resp.status_code, resp.text

    return fetch


@dataclass
class CrawlStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: int = 0
    blocked_by_robots: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"added={self.added} updated={self.updated} unchanged={self.unchanged} "
            f"removed={self.removed} failed={self.failed} blocked_by_robots={self.blocked_by_robots}"
        )


# ── Sitemaps ──────────────────────────────────────────────────────────────────

def parse_sitemap(xml_text: str) -> Tuple[List[str], List[str]]:
    """Return (page_urls, child_sitemap_urls) from a urlset or sitemapindex document."""
    root = ET.fromstring(xml_text)
    tag = root.tag.rsplit("}", 1)[-1]
    locs = [
        (el.text or "").strip()
        for el in root.iter()
        if el.tag.rsplit("}", 1)[-1] == "loc" and (el.text or "").strip()
    ]
    return ([], locs) if tag == "sitemapindex" else (locs, [])


def collect_urls(sitemap_url: str, fetch: Fetcher, depth: int = 0) -> List[str]:
    """Follow sitemap indexes (bounded depth) and return unique, same-host page URLs in order."""
    status, body = fetch(sitemap_url)
    if status != 200:
        raise RuntimeError(f"sitemap {sitemap_url} returned HTTP {status}")
    pages, children = parse_sitemap(body)
    if children and depth < MAX_SITEMAP_DEPTH:
        for child in children:
            pages.extend(collect_urls(child, fetch, depth + 1))
    host = urlparse(sitemap_url).netloc
    seen: Set[str] = set()
    out: List[str] = []
    for url in pages:
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https") and parsed.netloc == host and url not in seen:
            seen.add(url)
            out.append(url)
    return out


# ── Cleaning and chunking ─────────────────────────────────────────────────────

_NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe"]


def clean_html(html: str) -> Tuple[str, str]:
    """Return (title, main text). Drops navigation and boilerplate, keeps heading/paragraph breaks."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td"]):
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if text:
            blocks.append(text)
    return title, "\n\n".join(blocks)


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Paragraph-aware chunks of at most ~max_chars with a small tail overlap between chunks."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > max_chars:  # one huge paragraph: hard split on a space
            cut = para.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            piece, para = para[:cut], para[cut:].lstrip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + para) if tail else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def page_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


# ── Crawl ─────────────────────────────────────────────────────────────────────

def _robots_for(sitemap_url: str, fetch: Fetcher) -> urllib.robotparser.RobotFileParser:
    parsed = urlparse(sitemap_url)
    robots = urllib.robotparser.RobotFileParser()
    try:
        status, body = fetch(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        robots.parse(body.splitlines() if status == 200 else [])
    except Exception:
        robots.parse([])
    return robots


def _stored_pages(collection) -> dict:
    """url -> {"ids": [...], "hash": str, "count": int} for everything currently stored."""
    got = collection.get(include=["metadatas"])
    pages: dict = {}
    for chunk_id, meta in zip(got["ids"], got["metadatas"]):
        if not meta or "url" not in meta:
            continue
        entry = pages.setdefault(meta["url"], {"ids": [], "hash": meta.get("content_hash"), "count": meta.get("chunk_count")})
        entry["ids"].append(chunk_id)
    return pages


def crawl_site(
    sitemap_url: str,
    collection,
    fetch: Optional[Fetcher] = None,
    max_pages: int = 500,
    delay: float = 0.5,
    max_chars: int = 1200,
) -> CrawlStats:
    fetch = fetch or default_fetcher()
    stats = CrawlStats()
    urls = collect_urls(sitemap_url, fetch)[:max_pages]  # raises on failure -> nothing is deleted
    robots = _robots_for(sitemap_url, fetch)
    stored = _stored_pages(collection)
    gone: Set[str] = set()
    crawled_at = datetime.now(timezone.utc).isoformat()

    for i, url in enumerate(urls):
        if not robots.can_fetch(USER_AGENT, url):
            stats.blocked_by_robots += 1
            continue
        if i and delay:
            time.sleep(delay)
        try:
            status, html = fetch(url)
        except Exception as exc:  # network error: keep the old copy, report it
            stats.failed += 1
            stats.errors.append(f"{url}: {exc}")
            continue
        if status in (404, 410):
            gone.add(url)
            continue
        if status != 200:
            stats.failed += 1
            stats.errors.append(f"{url}: HTTP {status}")
            continue

        title, text = clean_html(html)
        chunks = chunk_text(text, max_chars=max_chars)
        if not chunks:
            stats.failed += 1
            stats.errors.append(f"{url}: no extractable text")
            continue
        digest = content_hash(text)
        existing = stored.get(url)
        if existing and existing["hash"] == digest and existing["count"] == len(chunks):
            stats.unchanged += 1
            continue
        if existing:
            collection.delete(ids=existing["ids"])  # old chunk count may differ from the new one
        pid = page_id(url)
        collection.upsert(
            ids=[f"{pid}#{n}" for n in range(len(chunks))],
            documents=[(f"{title}\n\n{c}" if title else c) for c in chunks],
            metadatas=[
                {"url": url, "title": title, "content_hash": digest, "chunk_index": n,
                 "chunk_count": len(chunks), "crawled_at": crawled_at}
                for n in range(len(chunks))
            ],
        )
        stats.updated += 1 if existing else 0
        stats.added += 0 if existing else 1

    # Pages that left the sitemap, or now 404/410, disappear from the index.
    in_sitemap = set(urls)
    for url, entry in stored.items():
        if url not in in_sitemap or url in gone:
            collection.delete(ids=entry["ids"])
            stats.removed += 1
    return stats
