"""
Market News Scraper
====================
Pulls top market-moving stories from Yahoo Finance, Google News RSS, and
MarketWatch RSS, filters for market relevance, deduplicates, and ranks
by a combination of recency and relevance.

Usage:
    python market_news.py              # top 5 stories
    python market_news.py -n 10        # top 10
    python market_news.py --json       # JSON output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── RSS feeds ────────────────────────────────────────────────────────────

GOOGLE_NEWS_FEEDS = [
    # Business topic
    ("https://news.google.com/rss/topics/"
     "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB"
     "?hl=en-US&gl=US&ceid=US:en"),
    # Targeted market searches
    "https://news.google.com/rss/search?q=stock+market+today+when:1d&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=S%26P+500+OR+Nasdaq+OR+Dow+Jones+when:1d&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Fed+rate+OR+inflation+OR+tariffs+when:1d&hl=en-US&gl=US&ceid=US:en",
]

YAHOO_RSS_URL = "https://finance.yahoo.com/news/rssindex"

MARKETWATCH_FEEDS = [
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
]

# ── relevance keywords (case-insensitive) ────────────────────────────────

# High-signal: these strongly indicate a market-moving story
HIGH_RELEVANCE = {
    "s&p", "s&p 500", "sp500", "nasdaq", "dow jones", "dow",
    "fed ", "federal reserve", "fomc", "rate cut", "rate hike",
    "inflation", "cpi", "ppi", "jobs report", "nonfarm", "unemployment",
    "gdp", "recession", "bear market", "bull market", "correction",
    "rally", "selloff", "sell-off", "crash", "plunge", "surge",
    "tariff", "trade war", "sanctions",
    "earnings", "revenue miss", "revenue beat", "guidance",
    "ipo", "merger", "acquisition", "buyout",
    "treasury", "yields", "bond", "10-year",
    "oil price", "crude", "gold price",
    "market today", "markets today", "wall street",
    "futures", "premarket", "after hours",
    "nvidia", "apple", "microsoft", "tesla", "amazon", "google",
    "meta", "magnificent seven", "mag 7",
}

# Medium-signal
MED_RELEVANCE = {
    "stock", "stocks", "shares", "equities", "index",
    "sector", "etf", "options", "volatility", "vix",
    "economy", "economic", "consumer", "spending", "retail sales",
    "bank", "banking", "crypto", "bitcoin",
    "analyst", "upgrade", "downgrade", "price target",
    "buyback", "dividend", "split",
}


@dataclass
class Story:
    title: str
    source: str
    url: str
    published: Optional[datetime]
    origin: str            # "yahoo" | "google" | "marketwatch"
    relevance: float = 0.0

    @property
    def age_label(self) -> str:
        if self.published is None:
            return "unknown"
        now = datetime.now(timezone.utc)
        delta = now - self.published
        mins = int(delta.total_seconds() / 60)
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{delta.days}d ago"


# ── scrapers ─────────────────────────────────────────────────────────────

def _parse_rss(url: str, origin: str, source_name: str = "") -> List[Story]:
    """Generic RSS parser."""
    stories = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        feed = feedparser.parse(resp.text)
        for entry in feed.entries:
            pub = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

            title = entry.get("title", "").strip()
            source = source_name

            # Google News titles end with " - SourceName"
            if origin == "google" and " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip()

            if not title:
                continue

            stories.append(Story(
                title=title,
                source=source or origin,
                url=entry.get("link", ""),
                published=pub,
                origin=origin,
            ))
    except Exception:
        pass
    return stories


def scrape_google_news() -> List[Story]:
    stories = []
    for url in GOOGLE_NEWS_FEEDS:
        stories.extend(_parse_rss(url, "google"))
    return stories


def scrape_yahoo_rss() -> List[Story]:
    return _parse_rss(YAHOO_RSS_URL, "yahoo", "Yahoo Finance")


def scrape_marketwatch() -> List[Story]:
    stories = []
    for url in MARKETWATCH_FEEDS:
        stories.extend(_parse_rss(url, "marketwatch", "MarketWatch"))
    return stories


def scrape_yahoo_page() -> List[Story]:
    """Scrape Yahoo Finance market news page as fallback."""
    stories = []
    try:
        resp = requests.get(
            "https://finance.yahoo.com/topic/stock-market-news/",
            headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        for item in soup.select("h3 a, h2 a"):
            title = item.get_text(strip=True)
            href = item.get("href", "")
            if not title or len(title) < 15:
                continue
            if href.startswith("/"):
                href = "https://finance.yahoo.com" + href
            stories.append(Story(
                title=title,
                source="Yahoo Finance",
                url=href,
                published=None,
                origin="yahoo",
            ))
    except Exception:
        pass
    return stories


# ── relevance scoring ────────────────────────────────────────────────────

SPAM_PATTERNS = {
    "presale", "airdrop", "meme coin", "shib-pattern", "token launch",
    "free ai powered", "year in review", "sponsored", "advertisement",
    "best stocks to buy", "best dividend stocks", "retirees",
}

SPAM_SOURCES = {
    "mfd.ru", "benzinga", "insidermonkey",
}


def _is_spam(story: Story) -> bool:
    title_lower = story.title.lower()
    src_lower = story.source.lower()
    for pat in SPAM_PATTERNS:
        if pat in title_lower:
            return True
    for src in SPAM_SOURCES:
        if src in src_lower:
            return True
    return False


def score_relevance(story: Story) -> float:
    """Score 0–1 based on how market-moving the headline looks."""
    if _is_spam(story):
        return 0.0

    title_lower = story.title.lower()
    score = 0.0

    for kw in HIGH_RELEVANCE:
        if kw in title_lower:
            score += 0.4
    for kw in MED_RELEVANCE:
        if kw in title_lower:
            score += 0.15

    # boost stories from market-focused sources
    src = story.source.lower()
    if any(s in src for s in ("cnbc", "bloomberg", "reuters", "wsj",
                               "wall street journal", "marketwatch",
                               "financial times", "barron",
                               "investor's business daily")):
        score += 0.3

    return min(score, 1.0)


# ── dedup & rank ─────────────────────────────────────────────────────────

def _normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def deduplicate(stories: List[Story]) -> List[Story]:
    """Remove near-duplicate headlines, keeping the best version."""
    seen = {}
    for s in stories:
        key = " ".join(_normalise(s.title).split()[:6])
        if key in seen:
            existing = seen[key]
            # prefer: has timestamp > higher relevance > first seen
            if (existing.published is None and s.published is not None) or \
               (s.relevance > existing.relevance):
                seen[key] = s
        else:
            seen[key] = s
    return list(seen.values())


def rank_stories(stories: List[Story]) -> List[Story]:
    """Rank by weighted combination of relevance and recency."""
    now = datetime.now(timezone.utc)

    def sort_key(s):
        # recency: hours old → 0–1 (1 = brand new, 0 = 24h+)
        if s.published:
            hours_old = (now - s.published).total_seconds() / 3600
            recency = max(0.0, 1.0 - hours_old / 24.0)
        else:
            recency = 0.0
        # 60% relevance, 40% recency
        return 0.6 * s.relevance + 0.4 * recency

    return sorted(stories, key=sort_key, reverse=True)


# ── main ─────────────────────────────────────────────────────────────────

def get_top_stories(n: int = 5) -> List[Story]:
    """Fetch from all sources, score, dedup, rank, return top N."""
    all_stories = []
    all_stories.extend(scrape_google_news())
    all_stories.extend(scrape_yahoo_rss())
    all_stories.extend(scrape_marketwatch())

    if len(all_stories) < n:
        all_stories.extend(scrape_yahoo_page())

    # score relevance
    for s in all_stories:
        s.relevance = score_relevance(s)

    # filter out clearly irrelevant stories
    all_stories = [s for s in all_stories if s.relevance > 0]

    deduped = deduplicate(all_stories)
    ranked = rank_stories(deduped)
    return ranked[:n]


def print_stories(stories: List[Story]) -> None:
    width = 60
    print("=" * width)
    print("TOP MARKET-MOVING STORIES")
    print("=" * width)
    for i, s in enumerate(stories, 1):
        print(f"\n  {i}. {s.title}")
        print(f"     {s.source}  |  {s.age_label}")
        print(f"     {s.url}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top market-moving stories")
    parser.add_argument("-n", type=int, default=5, help="Number of stories")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    stories = get_top_stories(args.n)

    if args.json:
        out = []
        for s in stories:
            d = asdict(s)
            d["published"] = s.published.isoformat() if s.published else None
            d["age"] = s.age_label
            out.append(d)
        print(json.dumps(out, indent=2))
    else:
        if not stories:
            print("No stories found. Check your network connection.")
            sys.exit(1)
        print_stories(stories)
