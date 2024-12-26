import requests
import json
import numpy as np

# For sentiment analysis:
# Option 1: Use Hugging Face Transformers pipeline
from transformers import pipeline

# OR
# Option 2: Use TextBlob (a simpler approach)
# from textblob import TextBlob

def fetch_ada_price():
    """
    Fetch current Cardano (ADA) price from a public API such as CoinGecko.
    Returns the current price in USD (float).
    """
    try:
        # CoinGecko example endpoint
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "cardano",
            "vs_currencies": "usd"
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        # Extract the price from the JSON
        current_price = data["cardano"]["usd"]
        return current_price
    except Exception as e:
        print(f"Error fetching ADA price: {e}")
        return None

def fetch_news_headlines():
    """
    Returns a list of sample headlines about ADA or Cardano.
    In a real agent, this would call a news API or parse an RSS feed.
    """
    # For demonstration, we will define some static headlines.
    # You can replace this with actual API calls or web scrapers.
    headlines = [
        "Cardano Surges After Positive Development Updates",
        "Experts Warn Of Possible Correction in Cardano",
        "Major Financial Institution to Begin Holding ADA Reserves",
        "Bearish Signals Emerge Despite Cardano's Strong Performance",
        "Investors Show Growing Interest in Cardano's Future"
    ]
    return headlines

def analyze_sentiment(headlines):
    """
    Uses a Hugging Face sentiment pipeline to analyze each headline.
    Returns the average sentiment score (range roughly from -1.0 to +1.0).
    """
    # Initialize sentiment pipeline (using a default pretrained model)
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    scores = []
    for headline in headlines:
        result = sentiment_pipeline(headline)[0]  # returns [{'label': 'POSITIVE'/'NEGATIVE', 'score': float}]
        label = result['label']
        score = result['score']
        
        # Convert the model's "label" into a numeric scale:
        # POSITIVE = +1, NEGATIVE = -1
        if label == "POSITIVE":
            numeric_score = score  # e.g., 0.95 -> strongly positive
        else:  # "NEGATIVE"
            numeric_score = -score # e.g., -0.90 -> strongly negative
        
        scores.append(numeric_score)
    
    # Calculate the average sentiment
    avg_sentiment = np.mean(scores)
    return avg_sentiment

def make_decision(avg_sentiment, current_price):
    """
    Given the average sentiment and the current ADA price,
    decide whether to 'Buy', 'Sell', or 'Hold'.
    
    A simple rule-based approach:
    - If sentiment > 0.3, recommend 'Buy'
    - If sentiment < -0.3, recommend 'Sell'
    - Otherwise, 'Hold'
    
    In a real system, you might incorporate price movements,
    technical indicators, or more sophisticated logic.
    """
    if avg_sentiment > 0.3:
        decision = "BUY"
    elif avg_sentiment < -0.3:
        decision = "SELL"
    else:
        decision = "HOLD"
    return decision

def run_agentic_ai():
    """
    Main function to run our minimal agentic AI for ADA.
    """
    print("=== Fetching ADA Price ===")
    price = fetch_ada_price()
    if price is None:
        print("Could not fetch price. Exiting.")
        return
    
    print(f"Current ADA Price (USD): {price}")
    
    print("\n=== Fetching News Headlines ===")
    headlines = fetch_news_headlines()
    for hl in headlines:
        print(" -", hl)
    
    print("\n=== Analyzing Sentiment ===")
    avg_sentiment = analyze_sentiment(headlines)
    print(f"Average sentiment score: {avg_sentiment:.4f}")
    
    print("\n=== Making Decision ===")
    decision = make_decision(avg_sentiment, price)
    print(f"Final Recommendation: {decision}\n")

if __name__ == "__main__":
    run_agentic_ai()

