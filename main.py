"""
Daily news agent: semiconductors, geopolitics, finance.

What it does, in order:
  1. Reads your RSS sources from feeds.yaml
  2. Keeps only articles published in the last ~30 hours
  3. Throws away anything it already showed you before (seen.json)
  4. Throws away near-duplicate stories (many outlets, same TSMC news)
  5. Asks a free AI model to score each story 1-5 for how much YOU care
  6. Builds a clean web page from the ones scoring 4+, grouped by topic
  7. Remembers what it showed, so tomorrow is fresh

You should not need to edit this file. Edit feeds.yaml and PROFILE below.
"""

import os
import re
import json
import html
import datetime as dt

import yaml
import feedparser
import requests


# ---------------------------------------------------------------------------
# EDIT THIS: describe yourself so the AI knows what is relevant to you.
# The more specific, the shorter and better your digest gets.
# ---------------------------------------------------------------------------
PROFILE = """
I follow the semiconductor industry, geopolitics, and financial markets.
I care most about: chip export controls and sanctions, TSMC/ASML/Nvidia/Intel/
Samsung, fab construction and capacity, US-China tech competition, Taiwan
security, AI compute demand, central bank rate decisions, and major market
moves. I am based in India, so India-specific policy and market news is
relevant too.

Geopolitics is my top priority and I want real depth on it: military
movements and conflicts, sanctions and export control changes, diplomatic
meetings and their outcomes, alliance shifts (NATO, Quad, BRICS), Taiwan
Strait tensions, South China Sea incidents, Russia-Ukraine and Middle East
developments, and any event that could move markets or chip supply chains.
For geopolitics specifically, give me the concrete details - who, what
happened, numbers/dates, and why it matters - not just a headline restated.

I do NOT care about: consumer gadget reviews, gaming GPU benchmarks, celebrity
business gossip, sports, routine stock-picking opinion pieces.
"""

# How many hours back to look. 30 gives a safety margin over a 24h cycle.
LOOKBACK_HOURS = 30

# Minimum relevance score (1-5) to include. 4 keeps only genuinely important
# stories, not borderline noise.
MIN_SCORE = 4

# Max articles sent to the AI per run. Keeps you inside free API limits.
MAX_TO_SCORE = 70

# Groq's free model. If this ever errors with "model not found", check
# https://console.groq.com/docs/models and paste a current name here.
MODEL = "llama-3.3-70b-versatile"

SEEN_FILE = "seen.json"
MAX_SEEN = 3000  # forget older entries so the file doesn't grow forever


# ---------------------------------------------------------------------------
# Step 1 + 2: fetch feeds, keep recent items
# ---------------------------------------------------------------------------
def fetch_articles():
    with open("feeds.yaml") as f:
        config = yaml.safe_load(f)

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=LOOKBACK_HOURS)
    articles = []
    errors = []

    for feed in config["feeds"]:
        try:
            parsed = feedparser.parse(feed["url"])
            if not parsed.entries:
                errors.append(f"{feed['name']}: no entries")
                continue

            for entry in parsed.entries[:40]:
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    when = dt.datetime(*published[:6], tzinfo=dt.timezone.utc)
                    if when < cutoff:
                        continue
                else:
                    when = dt.datetime.now(dt.timezone.utc)

                title = clean(entry.get("title", ""))
                link = entry.get("link", "")
                if not title or not link:
                    continue

                summary = clean(entry.get("summary", ""))[:400]

                articles.append({
                    "title": title,
                    "url": link,
                    "source": feed["name"],
                    "topic": feed["topic"],
                    "summary": summary,
                    "when": when,
                })
        except Exception as e:
            errors.append(f"{feed['name']}: {e}")

    articles.sort(key=lambda a: a["when"], reverse=True)
    return articles, errors


def clean(text):
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Step 3 + 4: drop already-seen and near-duplicate stories
# ---------------------------------------------------------------------------
def load_seen():
    try:
        with open(SEEN_FILE) as f:
            return json.load(f)
    except Exception:
        return {"urls": [], "fingerprints": []}


def save_seen(seen):
    seen["urls"] = seen["urls"][-MAX_SEEN:]
    seen["fingerprints"] = seen["fingerprints"][-MAX_SEEN:]
    with open(SEEN_FILE, "w") as f:
        json.dump(seen, f, indent=1)


STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "as", "at", "by",
    "with", "from", "is", "are", "was", "were", "it", "its", "be", "will",
    "says", "say", "new", "after", "over", "amid", "that", "this",
}


def keywords(title):
    """The meaningful words in a headline, used to spot the same story twice."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def same_story(words_a, words_b, threshold=0.55):
    """True if two headlines overlap enough to be the same underlying story."""
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    # Compare against the shorter headline, so a longer rewrite still matches
    return overlap / min(len(words_a), len(words_b)) >= threshold


def canonical(url):
    """Drop tracking junk so the same link isn't seen as two links."""
    return url.split("?")[0].split("#")[0].rstrip("/")


def deduplicate(articles, seen):
    seen_urls = set(seen["urls"])
    old_stories = [set(k) for k in seen["fingerprints"]]
    kept_stories = []
    fresh = []

    for a in articles:
        url_key = canonical(a["url"])
        if url_key in seen_urls:
            continue

        words = keywords(a["title"])
        if any(same_story(words, old) for old in old_stories):
            continue
        if any(same_story(words, kept) for kept in kept_stories):
            continue

        seen_urls.add(url_key)
        kept_stories.append(words)
        a["url_key"] = url_key
        a["words"] = sorted(words)
        fresh.append(a)

    return fresh


# ---------------------------------------------------------------------------
# Step 5: ask the AI to score relevance and write summaries
# ---------------------------------------------------------------------------
def score_articles(articles):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set. See README.")

    items = articles[:MAX_TO_SCORE]
    listing = "\n".join(
        f'{i}. [{a["topic"]}] {a["title"]} ({a["source"]}) - {a["summary"][:150]}'
        for i, a in enumerate(items)
    )

    prompt = f"""Here is a reader profile:
{PROFILE}

Below are today's news headlines. For EACH one, rate 1-5 how relevant it is to
this reader (5 = they must see this today, 1 = irrelevant noise) and write a
summary in your own words.

Scoring rules:
- Score down (2 or lower) anything speculative, opinion-driven, clickbait-
  phrased, or based on unnamed "sources say" rumors rather than confirmed
  facts. This reader wants accurate, confirmed developments, not chatter.
- Score up (4-5) hard news from primary or authoritative sources: official
  statements, government/agency releases, confirmed events, earnings/data
  releases, and significant on-the-ground developments.
- If two headlines describe the same event, judge them on substance, not
  on how dramatic the headline sounds.

Summary length depends on topic:
- geopolitics items: write 2-3 detailed sentences. Include concrete specifics
  from the snippet given - who is involved, what exactly happened, any
  numbers, dates or locations, and why it matters strategically. Stick to
  what the snippet actually states; do not speculate or add claims that
  aren't in the source snippet. Do not just reword the headline.
- semis and finance items: one clear factual sentence is enough.

Headlines:
{listing}

Reply with ONLY a JSON array, no markdown fences, no commentary:
[{{"i": 0, "score": 4, "line": "summary text"}}, ...]
Include an entry for every headline."""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 6000,
        },
        timeout=120,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]

    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            print("AI reply could not be parsed. Showing unscored digest.")
            for a in items:
                a["score"], a["line"] = 3, a["summary"][:200]
            return items
        scores = json.loads(match.group())

    for entry in scores:
        idx = entry.get("i")
        if isinstance(idx, int) and 0 <= idx < len(items):
            items[idx]["score"] = int(entry.get("score", 3))
            items[idx]["line"] = entry.get("line", "")

    for a in items:
        a.setdefault("score", 3)
        a.setdefault("line", a["summary"][:200])

    return items


# ---------------------------------------------------------------------------
# Step 6: build the web page
# ---------------------------------------------------------------------------
TOPIC_LABELS = {
    "semis": "Semiconductors",
    "geopolitics": "Geopolitics",
    "finance": "Finance &amp; Markets",
}

PAGE_STYLE = """
<style>
  :root {
    --bg: #f7f7f5; --card: #ffffff; --text: #1a1a1a; --muted: #666;
    --border: #e5e5e5; --accent: #0b57d0; --star: #e8a33d;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #15161a; --card: #1e1f24; --text: #eaeaea; --muted: #999;
             --border: #333; --accent: #7db3ff; --star: #f0b93d; }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px 16px 60px; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.55;
  }
  .wrap { max-width: 700px; margin: 0 auto; }
  h1 { font-size: 24px; margin: 0 0 2px; }
  .updated { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
  h2 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--muted); border-bottom: 1px solid var(--border);
    padding-bottom: 8px; margin: 32px 0 16px;
  }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px 18px; margin-bottom: 14px;
  }
  .card a {
    color: var(--text); text-decoration: none; font-weight: 600; font-size: 16px;
    display: block; margin-bottom: 6px;
  }
  .card a:hover { color: var(--accent); }
  .card .line { font-size: 14.5px; color: var(--text); opacity: 0.9; }
  .card .meta { font-size: 12px; color: var(--muted); margin-top: 8px; }
  .stars { color: var(--star); }
  .empty { color: var(--muted); padding: 40px 0; text-align: center; }
  .footer { color: var(--muted); font-size: 11px; margin-top: 40px; text-align: center; }
</style>
"""


def build_page(articles, errors):
    now = dt.datetime.now(dt.timezone.utc)
    updated = now.strftime("%A, %d %B %Y - %H:%M UTC")
    keepers = [a for a in articles if a["score"] >= MIN_SCORE]
    keepers.sort(key=lambda a: a["score"], reverse=True)

    body = [f"""<h1>Daily Brief</h1>
<div class="updated">Updated {updated} &middot; {len(keepers)} stories</div>"""]

    if not keepers:
        body.append('<div class="empty">Nothing met the relevance bar in this run.</div>')

    for topic in ["geopolitics", "semis", "finance"]:
        group = [a for a in keepers if a["topic"] == topic]
        if not group:
            continue
        body.append(f'<h2>{TOPIC_LABELS[topic]}</h2>')
        for a in group:
            stars = "&#9733;" * a["score"]
            body.append(f"""<div class="card">
<a href="{html.escape(a['url'])}" target="_blank" rel="noopener">{html.escape(a['title'])}</a>
<div class="line">{html.escape(a['line'])}</div>
<div class="meta">{html.escape(a['source'])} &middot; <span class="stars">{stars}</span></div>
</div>""")

    if errors:
        body.append(f'<div class="footer">Feed issues this run: {html.escape(", ".join(errors[:5]))}</div>')

    body.append('<div class="footer">Refreshes automatically once a day.</div>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Daily Brief</title>
{PAGE_STYLE}
</head>
<body>
<div class="wrap">
{''.join(body)}
</div>
</body>
</html>"""


def save_page(html_content):
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(html_content)
    print("Wrote docs/index.html")


# ---------------------------------------------------------------------------
def main():
    print("Fetching feeds...")
    articles, errors = fetch_articles()
    print(f"  {len(articles)} recent articles, {len(errors)} feed errors")

    seen = load_seen()
    fresh = deduplicate(articles, seen)
    print(f"  {len(fresh)} after removing seen and duplicate stories")

    if not fresh:
        print("Nothing new. Leaving yesterday's page as-is.")
        return

    print("Scoring with AI...")
    scored = score_articles(fresh)

    page_html = build_page(scored, errors)
    save_page(page_html)

    seen["urls"].extend(a["url_key"] for a in scored)
    seen["fingerprints"].extend(a["words"] for a in scored)
    save_seen(seen)
    print("Done.")


if __name__ == "__main__":
    main()
