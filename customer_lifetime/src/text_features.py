from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from textblob import TextBlob

NEGATION_RE = re.compile(r"\b(not|never|no|can't|cannot|won't|wouldn't|don't)\b")
CHURN_RE = re.compile(
    r"\b(cancel|cancell|switching|alternative|competitor|leaving|downgrad|refund)\b"
)
PRAISE_RE = re.compile(
    r"\b(love|excellent|amazing|best|recommend|worth|outstanding|perfect)\b"
)
TOKEN_RE = re.compile(r"\b[a-z]{3,}\b")


def clean_text(text: str | None) -> str:
    if pd.isna(text) or text is None:
        return ""
    value = str(text).lower()
    value = re.sub(r"\.{2,}", ".", value)
    value = re.sub(r"[^a-z0-9\s\.\,\!\?']", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: str | None) -> list[str]:
    return TOKEN_RE.findall(clean_text(text))


def extract_sentiment_features(text: str | None) -> dict[str, float | int]:
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "sentiment_polarity": 0.0,
            "sentiment_subjectivity": 0.0,
            "has_negation": 0,
            "churn_language": 0,
            "praise_language": 0,
            "feedback_word_count": 0,
        }

    blob = TextBlob(cleaned)
    return {
        "sentiment_polarity": float(blob.sentiment.polarity),
        "sentiment_subjectivity": float(blob.sentiment.subjectivity),
        "has_negation": int(bool(NEGATION_RE.search(cleaned))),
        "churn_language": int(bool(CHURN_RE.search(cleaned))),
        "praise_language": int(bool(PRAISE_RE.search(cleaned))),
        "feedback_word_count": len(cleaned.split()),
    }


def top_words_by_mask(
    feedback: pd.Series,
    mask: pd.Series,
    top_n: int = 15,
) -> list[tuple[str, int]]:
    counter = Counter()
    for text in feedback[mask]:
        counter.update(tokenize(text))
    return counter.most_common(top_n)

