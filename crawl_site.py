"""
crawl_site.py — Crawl a public docs site into the `crawled_docs` ChromaDB collection.

Usage:
    python crawl_site.py --sitemap https://docs.example.com/sitemap.xml
    python crawl_site.py --sitemap https://docs.example.com/sitemap.xml --every-hours 24   # nightly loop

Or run it from cron instead of --every-hours:
    0 2 * * *  cd /path/to/bug_triage_agent && .venv/bin/python crawl_site.py --sitemap <url>
"""

import argparse
import time

from dotenv import load_dotenv

load_dotenv()

from agent.crawler import crawl_site
from agent.vectorstore import CRAWLED_COLLECTION_NAME, _get_collection


def main():
    parser = argparse.ArgumentParser(description="Crawl a sitemap into ChromaDB")
    parser.add_argument("--sitemap", required=True, help="URL of sitemap.xml (or a sitemap index)")
    parser.add_argument("--collection", default=CRAWLED_COLLECTION_NAME)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between page requests")
    parser.add_argument("--every-hours", type=float, default=0, help="repeat forever at this interval (0 = run once)")
    args = parser.parse_args()

    while True:
        stats = crawl_site(args.sitemap, _get_collection(args.collection), max_pages=args.max_pages, delay=args.delay)
        print(f"Crawl finished: {stats.summary()}")
        for err in stats.errors:
            print(f"  ! {err}")
        if not args.every_hours:
            break
        time.sleep(args.every_hours * 3600)


if __name__ == "__main__":
    main()
