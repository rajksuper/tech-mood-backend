from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]
    label = result['label'].lower()
    score = result['score']

    if label == "positive":
        mapped = score
    elif label == "negative":
        mapped = -score
    else:
        mapped = 0  # neutral

    return label, mapped, result
