"""
Daily news agent: semiconductors, geopolitics, finance.

What it does, in order:
  1. Reads your RSS sources from feeds.yaml
  2. Keeps only articles published in the last ~30 hours
  3. Throws away anything it already emailed you before (seen.json)
  4. Throws away near-duplicate stories (30 outlets, same TSMC news)
  5. Asks a free AI model to score each story 1-5 for how much YOU care
  6. Emails you the ones scoring 3+, grouped by topic
  7. Remembers what it sent, so tomorrow is fresh

You should not need to edit this file. Edit feeds.yaml and PROFILE below.
"""

import os
import re
import json
import html
import smtplib
import datetime as dt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
I do NOT care about: consumer gadget reviews, gaming GPU benchmarks, celebrity
business gossip, sports, routine stock-picking opinion pieces.
"""

# How many hours back to look. 30 gives a safety margin over a 24h cycle.
LOOKBACK_HOURS = 30

# Minimum relevance score (1-5) to include in the email.
MIN_SCORE = 3

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
    # Headlines already emailed on previous days
    old_stories = [set(k) for k in seen["fingerprints"]]
    kept_stories = []
    fresh = []

    for a in articles:
        url_key = canonical(a["url"])
        if url_key in seen_urls:
            continue

        words = keywords(a["title"])
        # Same story as something sent before, or as something earlier in this run?
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
# Step 5: ask the AI to score relevance and write one-line summaries
# ---------------------------------------------------------------------------
def score_articles(articles):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set. See README step 3.")

    items = articles[:MAX_TO_SCORE]
    listing = "\n".join(
        f'{i}. [{a["topic"]}] {a["title"]} ({a["source"]}) - {a["summary"][:150]}'
        for i, a in enumerate(items)
    )

    prompt = f"""Here is a reader profile:
{PROFILE}

Below are today's news headlines. For EACH one, rate 1-5 how relevant it is to
this reader (5 = they must see this today, 1 = irrelevant noise) and write a
single factual sentence summarising it in your own words.

Headlines:
{listing}

Reply with ONLY a JSON array, no markdown fences, no commentary:
[{{"i": 0, "score": 4, "line": "one sentence"}}, ...]
Include an entry for every headline."""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 4000,
        },
        timeout=120,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]

    # Strip code fences if the model adds them anyway
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            print("AI reply could not be parsed. Sending unscored digest.")
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
# Step 6: build and send the email
# ---------------------------------------------------------------------------
TOPIC_LABELS = {
    "semis": "Semiconductors",
    "geopolitics": "Geopolitics",
    "finance": "Finance &amp; Markets",
}


def build_email(articles, errors):
    today = dt.datetime.now().strftime("%A, %d %B %Y")
    keepers = [a for a in articles if a["score"] >= MIN_SCORE]
    keepers.sort(key=lambda a: a["score"], reverse=True)

    parts = [f"""<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;
max-width:640px;margin:0 auto;color:#1a1a1a;line-height:1.5">
<h1 style="font-size:20px;margin:0 0 4px">Daily Brief</h1>
<div style="color:#666;font-size:13px;margin-bottom:24px">{today} &middot;
{len(keepers)} stories</div>"""]

    if not keepers:
        parts.append("<p>Nothing met your relevance bar today. Quiet news cycle, "
                     "or your filters are tight.</p>")

    for topic in ["semis", "geopolitics", "finance"]:
        group = [a for a in keepers if a["topic"] == topic]
        if not group:
            continue
        parts.append(f"""<h2 style="font-size:14px;text-transform:uppercase;
letter-spacing:0.06em;color:#888;border-bottom:1px solid #e5e5e5;
padding-bottom:6px;margin:28px 0 14px">{TOPIC_LABELS[topic]}</h2>""")

        for a in group:
            stars = "&#9733;" * a["score"]
            parts.append(f"""<div style="margin-bottom:18px">
<a href="{html.escape(a['url'])}" style="color:#0b57d0;text-decoration:none;
font-weight:600;font-size:15px">{html.escape(a['title'])}</a>
<div style="font-size:14px;color:#333;margin-top:3px">{html.escape(a['line'])}</div>
<div style="font-size:12px;color:#999;margin-top:3px">{html.escape(a['source'])}
&middot; <span style="color:#e8a33d">{stars}</span></div></div>""")

    if errors:
        parts.append(f"""<div style="font-size:11px;color:#bbb;margin-top:32px;
border-top:1px solid #eee;padding-top:8px">Feed issues: {html.escape(', '.join(errors[:5]))}</div>""")

    parts.append("</div>")
    return "".join(parts), len(keepers)


def send_email(body, count):
    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")
    to = os.environ.get("EMAIL_TO", user)

    if not user or not password:
        print("Email not configured. Printing digest instead:\n")
        print(body)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Daily Brief - {count} stories - {dt.datetime.now():%d %b}"
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, password)
        server.send_message(msg)

    print(f"Sent {count} stories to {to}")


# ---------------------------------------------------------------------------
def main():
    print("Fetching feeds...")
    articles, errors = fetch_articles()
    print(f"  {len(articles)} recent articles, {len(errors)} feed errors")

    seen = load_seen()
    fresh = deduplicate(articles, seen)
    print(f"  {len(fresh)} after removing seen and duplicate stories")

    if not fresh:
        print("Nothing new. Not sending an email.")
        return

    print("Scoring with AI...")
    scored = score_articles(fresh)

    body, count = build_email(scored, errors)
    send_email(body, count)

    # Only remember what we actually scored, so unscored leftovers can appear tomorrow
    seen["urls"].extend(a["url_key"] for a in scored)
    seen["fingerprints"].extend(a["words"] for a in scored)
    save_seen(seen)
    print("Done.")


if __name__ == "__main__":
    main()
