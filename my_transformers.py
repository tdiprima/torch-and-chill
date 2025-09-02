# Transformers: Sentiment analysis with a pre-trained model
from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis"
)  # Loads a default model like distilbert
text = "I love using Hugging Face for AI!"
result = sentiment_pipeline(text)
print(
    "Sentiment result:", result
)  # Output: [{'label': 'POSITIVE', 'score': 0.9991549253463745}]
