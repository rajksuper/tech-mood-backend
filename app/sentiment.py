from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Keywords that strongly indicate sentiment (override low confidence)
POSITIVE_KEYWORDS = [
    "breakthrough", "surges", "soars", "wins", "launches", "raises", "funding",
    "record", "milestone", "success", "approved", "celebrates", "award", "profit",
    "growth", "bullish", "optimistic", "innovation", "partnership", "deal"
]

NEGATIVE_KEYWORDS = [
    "convicted", "arrested", "fraud", "hack", "breach", "layoffs", "lawsuit",
    "crash", "dies", "dead", "killed", "fails", "bankrupt", "scandal", "fired",
    "plunges", "drops", "bearish", "warns", "threat", "violation", "charged"
]


def analyze_sentiment(text):
    """
    Analyze sentiment with improved calibration for news headlines.
    
    Returns: (label, score, raw_result)
    - label: 'positive', 'negative', 'neutral', or 'mixed'
    - score: float from -1 to 1
    - raw_result: original model output
    """
    text_lower = text.lower()
    result = sentiment_model(text[:512])[0]
    label = result['label'].lower()
    score = result['score']
    
    # Check for strong keyword signals
    has_positive_keyword = any(kw in text_lower for kw in POSITIVE_KEYWORDS)
    has_negative_keyword = any(kw in text_lower for kw in NEGATIVE_KEYWORDS)
    
    # If keywords conflict, it's genuinely mixed
    if has_positive_keyword and has_negative_keyword:
        return "mixed", 0, result
    
    # Keyword override for low-confidence predictions
    if score < 0.6:
        if has_negative_keyword and not has_positive_keyword:
            return "negative", -0.6, result
        elif has_positive_keyword and not has_negative_keyword:
            return "positive", 0.6, result
    
    # Lower threshold for clear sentiment (0.5 instead of 0.7)
    # News headlines are formal, so model confidence is naturally lower
    if score < 0.5:
        # Very low confidence = genuinely neutral or mixed
        if label == "neutral":
            return "neutral", 0, result
        else:
            return "mixed", 0, result
    
    # Score between 0.5-0.6: lean toward the prediction but mark as mixed if neutral
    if score < 0.6:
        if label == "neutral":
            return "neutral", 0, result
        # Weak but present sentiment
        elif label == "positive":
            return "positive", score * 0.8, result  # Slightly discount weak positive
        elif label == "negative":
            return "negative", -score * 0.8, result
    
    # Score >= 0.6: trust the model
    if label == "positive":
        return "positive", score, result
    elif label == "negative":
        return "negative", -score, result
    else:
        return "neutral", 0, result