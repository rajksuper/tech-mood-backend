from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]
    label = result['label'].lower()
    score = result['score']

    # Flag low-confidence predictions as "mixed"
    if score < 0.7:
        label = "mixed"
        mapped = 0
    elif label == "positive":
        mapped = score
    elif label == "negative":
        mapped = -score
    else:  # neutral
        mapped = 0

    return label, mapped, result