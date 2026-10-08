import pytest

from agent.crawler import (
    chunk_text, clean_html, collect_urls, content_hash, crawl_site, parse_sitemap,
)

BASE = "https://docs.example.com"


def page(title, body, nav="Home | Pricing | Login"):
    return (
        f"<html><head><title>{title}</title><script>var x=1;</script></head><body>"
        f"<nav>{nav}</nav><main>{body}</main><footer>Copyright</footer></body></html>"
    )


def sitemap(*urls):
    entries = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return f'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{entries}</urlset>'


class FakeSite:
    """In-memory site: path -> (status, body). Records every request."""

    def __init__(self, pages):
        self.pages = dict(pages)
        self.requests = []

    def __call__(self, url):
        self.requests.append(url)
        return self.pages.get(url, (404, ""))


def make_site():
    return FakeSite({
        f"{BASE}/sitemap.xml": (200, sitemap(f"{BASE}/pool", f"{BASE}/lag")),
        f"{BASE}/robots.txt": (200, "User-agent: *\nAllow: /\n"),
        f"{BASE}/pool": (200, page("Redis pools", "<h1>Redis pools</h1><p>Release connections with context managers.</p>")),
        f"{BASE}/lag": (200, page("Kafka lag", "<h1>Kafka lag</h1><p>Move slow HTTP calls off the consumer thread.</p>")),
    })


def test_parse_urlset_and_index():
    assert parse_sitemap(sitemap("https://a/x", "https://a/y")) == (["https://a/x", "https://a/y"], [])
    index = '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://a/s1.xml</loc></sitemap></sitemapindex>'
    assert parse_sitemap(index) == ([], ["https://a/s1.xml"])


def test_collect_urls_follows_index_and_drops_other_hosts():
    site = FakeSite({
        f"{BASE}/index.xml": (200, '<sitemapindex xmlns="x"><sitemap><loc>%s/s1.xml</loc></sitemap></sitemapindex>' % BASE),
        f"{BASE}/s1.xml": (200, sitemap(f"{BASE}/a", "https://evil.example.net/b", "ftp://docs.example.com/c", f"{BASE}/a")),
    })
    assert collect_urls(f"{BASE}/index.xml", site) == [f"{BASE}/a"]


def test_collect_urls_raises_on_bad_status():
    with pytest.raises(RuntimeError):
        collect_urls(f"{BASE}/sitemap.xml", FakeSite({}))


def test_clean_html_drops_boilerplate_and_keeps_title():
    title, text = clean_html(page("Redis pools", "<h1>Pools</h1><p>Use  a   context manager.</p><ul><li>one</li></ul>"))
    assert title == "Redis pools"
    assert "Use a context manager." in text and "one" in text
    assert "Pricing" not in text and "Copyright" not in text and "var x" not in text


def test_chunk_text_respects_limit_and_overlaps():
    text = "\n\n".join(f"paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, max_chars=500, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 60 + 2 for c in chunks)
    assert chunks[0][-30:] in chunks[1]  # tail of one chunk reappears at the head of the next


def test_chunk_text_splits_one_giant_paragraph():
    chunks = chunk_text("word " * 1000, max_chars=300, overlap=0)
    assert len(chunks) > 10 and all(len(c) <= 300 for c in chunks)


def test_first_crawl_adds_every_page(collection):
    stats = crawl_site(f"{BASE}/sitemap.xml", collection, make_site(), delay=0)
    assert (stats.added, stats.updated, stats.unchanged, stats.removed) == (2, 0, 0, 0)
    got = collection.get(include=["metadatas"])
    assert {m["url"] for m in got["metadatas"]} == {f"{BASE}/pool", f"{BASE}/lag"}


def test_recrawl_with_no_changes_skips_everything(collection):
    site = make_site()
    crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    before = collection.get(include=["metadatas"])["metadatas"]
    stats = crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    assert (stats.added, stats.updated, stats.unchanged) == (0, 0, 2)
    assert collection.get(include=["metadatas"])["metadatas"] == before  # not re-embedded or rewritten


def test_changed_page_replaces_old_chunks(collection):
    site = make_site()
    crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    long_body = "<h1>Redis pools</h1>" + "".join(f"<p>{'pool tuning detail ' * 30} {i}</p>" for i in range(8))
    site.pages[f"{BASE}/pool"] = (200, page("Redis pools", long_body))
    stats = crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0, max_chars=600)
    assert (stats.updated, stats.unchanged) == (1, 1)
    chunks = collection.get(where={"url": f"{BASE}/pool"}, include=["metadatas"])
    assert len(chunks["ids"]) > 1
    assert {m["chunk_count"] for m in chunks["metadatas"]} == {len(chunks["ids"])}
    assert len(set(chunks["ids"])) == len(chunks["ids"])


def test_shrinking_page_leaves_no_stale_chunks(collection):
    site = make_site()
    long_body = "".join(f"<p>{'filler sentence ' * 30} {i}</p>" for i in range(8))
    site.pages[f"{BASE}/pool"] = (200, page("Redis pools", long_body))
    crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0, max_chars=600)
    assert len(collection.get(where={"url": f"{BASE}/pool"})["ids"]) > 1
    site.pages[f"{BASE}/pool"] = (200, page("Redis pools", "<p>Short now.</p>"))
    crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0, max_chars=600)
    assert len(collection.get(where={"url": f"{BASE}/pool"})["ids"]) == 1


def test_page_removed_from_sitemap_is_deleted(collection):
    site = make_site()
    crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    site.pages[f"{BASE}/sitemap.xml"] = (200, sitemap(f"{BASE}/pool"))
    stats = crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    assert stats.removed == 1
    assert {m["url"] for m in collection.get(include=["metadatas"])["metadatas"]} == {f"{BASE}/pool"}


def test_page_returning_404_is_deleted(collection):
    site = make_site()
    crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    site.pages[f"{BASE}/lag"] = (404, "")
    stats = crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    assert stats.removed == 1 and stats.failed == 0
    assert collection.get(where={"url": f"{BASE}/lag"})["ids"] == []


def test_transient_failure_keeps_old_copy(collection):
    site = make_site()
    crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    site.pages[f"{BASE}/lag"] = (503, "")
    stats = crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    assert stats.failed == 1 and stats.removed == 0
    assert collection.get(where={"url": f"{BASE}/lag"})["ids"]


def test_sitemap_outage_deletes_nothing(collection):
    site = make_site()
    crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    count = collection.count()
    site.pages[f"{BASE}/sitemap.xml"] = (500, "")
    with pytest.raises(RuntimeError):
        crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    assert collection.count() == count


def test_robots_disallow_is_honoured(collection):
    site = make_site()
    site.pages[f"{BASE}/robots.txt"] = (200, "User-agent: *\nDisallow: /lag\n")
    stats = crawl_site(f"{BASE}/sitemap.xml", collection, site, delay=0)
    assert stats.added == 1 and stats.blocked_by_robots == 1
    assert f"{BASE}/lag" not in site.requests


def test_content_hash_is_stable():
    assert content_hash("a") == content_hash("a") != content_hash("b")
