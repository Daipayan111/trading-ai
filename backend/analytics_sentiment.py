"""
Advanced Sentiment Analysis Module
====================================
Layered sentiment pipeline:
  Layer 1: FinBERT (Financial BERT) — if transformers available
  Layer 2: VADER Sentiment — fast, general purpose
  Layer 3: Keyword Matching (original fallback)

Also computes:
  - Aggregate sentiment from multiple news articles
  - Confidence score of the sentiment
  - Key headline highlights
"""
import warnings
warnings.filterwarnings('ignore')

# Flags for optional heavy libraries
_HAS_FINBERT = False
_HAS_VADER = False

try:
    from transformers import pipeline as hf_pipeline
    _FINBERT_PIPELINE = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        max_length=512,
        truncation=True,
    )
    _HAS_FINBERT = True
except Exception:
    _FINBERT_PIPELINE = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except Exception:
    _VADER = None


# =============================================================
# FINBERT SENTIMENT
# =============================================================
def _finbert_sentiment(headlines: list[str]) -> dict:
    """Run FinBERT on headlines and aggregate results."""
    if not _HAS_FINBERT or not headlines:
        return None

    try:
        scores = {"positive": 0, "negative": 0, "neutral": 0}
        for text in headlines[:10]:  # FinBERT is slow — cap at 10
            result = _FINBERT_PIPELINE(text[:512])[0]
            label = result['label'].lower()
            score = result['score']
            if label in scores:
                scores[label] += score

        total = sum(scores.values()) or 1
        dominant = max(scores, key=scores.get)
        confidence = round(scores[dominant] / total * 100, 1)

        label_map = {"positive": "BULLISH", "negative": "BEARISH", "neutral": "NEUTRAL"}
        score_map = {"positive": 1, "negative": -1, "neutral": 0}

        return {
            "method": "FinBERT",
            "sentiment": label_map.get(dominant, "NEUTRAL"),
            "confidence": confidence,
            "raw_scores": {k: round(v / total, 3) for k, v in scores.items()},
            "numeric_score": round(score_map.get(dominant, 0) * confidence / 100, 3),
        }
    except Exception as e:
        return None


# =============================================================
# VADER SENTIMENT
# =============================================================
def _vader_sentiment(headlines: list[str]) -> dict:
    """Run VADER on headlines and aggregate compound scores."""
    if not _HAS_VADER or not headlines:
        return None

    try:
        compounds = []
        for text in headlines[:15]:
            scores = _VADER.polarity_scores(text)
            compounds.append(scores['compound'])

        avg_compound = sum(compounds) / len(compounds) if compounds else 0

        if avg_compound >= 0.15:
            sentiment = "BULLISH"
        elif avg_compound <= -0.15:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        confidence = round(min(abs(avg_compound) * 100, 100), 1)

        return {
            "method": "VADER",
            "sentiment": sentiment,
            "confidence": confidence,
            "compound_score": round(avg_compound, 4),
            "numeric_score": round(avg_compound, 4),
        }
    except Exception:
        return None


# =============================================================
# KEYWORD FALLBACK
# =============================================================
def _keyword_sentiment(headlines: list[str]) -> dict:
    """Original keyword-based sentiment (always available as fallback)."""
    BULLISH_WORDS = [
        'growth', 'profit', 'jump', 'surge', 'buy', 'upgrade', 'high', 'win',
        'beat', 'record', 'dividend', 'expand', 'rally', 'gain', 'rise',
        'positive', 'strong', 'breakthrough', 'outperform', 'upbeat', 'bullish'
    ]
    BEARISH_WORDS = [
        'loss', 'fall', 'drop', 'sell', 'downgrade', 'low', 'miss', 'penalty',
        'lawsuit', 'cut', 'slump', 'crash', 'decline', 'weak', 'negative',
        'disappoint', 'bearish', 'warning', 'debt', 'default', 'risk'
    ]

    bullish = 0
    bearish = 0
    for h in headlines[:15]:
        text = h.lower()
        for w in BULLISH_WORDS:
            if w in text:
                bullish += 1
        for w in BEARISH_WORDS:
            if w in text:
                bearish += 1

    net = bullish - bearish
    total = bullish + bearish or 1
    confidence = round(abs(net) / total * 100, 1)

    if net > 1:
        sentiment = "BULLISH"
    elif net < -1:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    return {
        "method": "Keyword",
        "sentiment": sentiment,
        "confidence": confidence,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "numeric_score": round(net / total, 3),
    }


# =============================================================
# MAIN SENTIMENT ANALYZER
# =============================================================
def analyze_advanced_sentiment(ticker) -> dict:
    """
    Multi-layer sentiment analysis.
    Tries FinBERT → VADER → Keyword in order of preference.
    Returns final aggregated sentiment with source info.
    """
    try:
        news = ticker.news if hasattr(ticker, 'news') else []
        if not news:
            return _empty_sentiment()

        headlines = []
        for article in news[:15]:
            title = article.get('title', '')
            if title:
                headlines.append(title)

        if not headlines:
            return _empty_sentiment()

        # --- Try FinBERT first ---
        result = _finbert_sentiment(headlines)

        # --- Try VADER ---
        if result is None:
            result = _vader_sentiment(headlines)

        # --- Keyword fallback ---
        if result is None:
            result = _keyword_sentiment(headlines)

        # Also compute keyword alongside for comparison
        kw_result = _keyword_sentiment(headlines)

        # Build response
        sentiment = result.get('sentiment', 'NEUTRAL')
        numeric = result.get('numeric_score', 0)
        confidence = result.get('confidence', 0)

        return {
            "sentiment": sentiment,
            "sentiment_score": numeric,
            "confidence": confidence,
            "method_used": result.get('method', 'Unknown'),
            "finbert_available": _HAS_FINBERT,
            "vader_available": _HAS_VADER,
            "articles_analyzed": len(headlines),
            "top_headlines": headlines[:5],
            "keyword_cross_check": kw_result.get('sentiment', 'NEUTRAL'),
            "bullish_news": kw_result.get('bullish_count', 0),
            "bearish_news": kw_result.get('bearish_count', 0),
        }

    except Exception as e:
        return _empty_sentiment()


def _empty_sentiment() -> dict:
    return {
        "sentiment": "NEUTRAL",
        "sentiment_score": 0,
        "confidence": 0,
        "method_used": "No data",
        "finbert_available": _HAS_FINBERT,
        "vader_available": _HAS_VADER,
        "articles_analyzed": 0,
        "top_headlines": [],
        "keyword_cross_check": "NEUTRAL",
        "bullish_news": 0,
        "bearish_news": 0,
    }
