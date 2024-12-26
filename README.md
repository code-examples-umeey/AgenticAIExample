---

# Agentic AI for ADA (Cardano) News & Price Analysis
This repository showcases a **minimal agentic AI** that:
1. Fetches current Cardano (ADA) prices from a public API.
2. Retrieves headlines (in this demo, static examples).
3. Performs sentiment analysis on the headlines using a Hugging Face sentiment model.
4. Produces a **Buy**, **Sell**, or **Hold** recommendation based on sentiment.

> **Important**: This code is for demonstration purposes only and is **not** financial advice.

---

## Features

- **Fetch Real-Time Prices**: Uses the [CoinGecko API](https://www.coingecko.com/en/api/documentation) to get the current market price of Cardano (ADA).  
- **Simple Sentiment Analysis**: Analyzes headlines using the [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model.  
- **Rule-Based Decision**: Provides a basic Buy/Sell/Hold recommendation based on an average sentiment threshold.

---

## Getting Started

### Prerequisites

1. **Python 3.7+**  
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
   A minimal `requirements.txt` could look like:
   ```txt
   requests
   numpy
   transformers
   torch
   ```

3. (Optional) To run Hugging Face sentiment analysis on the GPU, ensure you have a compatible GPU and appropriate CUDA drivers installed.

### Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/YourUsername/agentic-ada-ai.git
   cd agentic-ada-ai
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the agent**:
   ```bash
   python main.py
   ```
   (Assuming the main script in this repo is named `main.py`; adjust accordingly.)

---

## Usage

1. **Run the Script**:  
   After installing dependencies, simply run:
   ```bash
   python main.py
   ```
2. **Output**:  
   - The script will print the current ADA price.  
   - It will analyze sentiment on the given headlines.  
   - Finally, it will print a recommendation (**BUY**, **SELL**, or **HOLD**).

Example console output might look like:

```
=== Fetching ADA Price ===
Current ADA Price (USD): 0.3182

=== Fetching News Headlines ===
 - Cardano Surges After Positive Development Updates
 - Experts Warn Of Possible Correction in Cardano
 - Major Financial Institution to Begin Holding ADA Reserves
 - Bearish Signals Emerge Despite Cardano's Strong Performance
 - Investors Show Growing Interest in Cardano's Future

=== Analyzing Sentiment ===
Average sentiment score: 0.3456

=== Making Decision ===
Final Recommendation: BUY
```

---

## Code Explanation

The core functions include:

1. **`fetch_ada_price()`**  
   - Calls the CoinGecko API to get ADA’s price in USD.

2. **`fetch_news_headlines()`**  
   - Returns a static list of sample headlines (replace with actual news sources in production).

3. **`analyze_sentiment(headlines)`**  
   - Uses a Hugging Face pipeline to get a **POSITIVE** or **NEGATIVE** score for each headline.  
   - Converts these to numeric values and averages them.

4. **`make_decision(avg_sentiment, current_price)`**  
   - Applies a simple rule-based threshold to output **BUY**, **SELL**, or **HOLD**.

5. **`run_agentic_ai()`**  
   - Orchestrates the full process (fetch price -> fetch news -> analyze -> decide).

---

## Demo Code

```python
import requests
import json
import numpy as np
from transformers import pipeline

def fetch_ada_price():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "cardano", "vs_currencies": "usd"}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        current_price = data["cardano"]["usd"]
        return current_price
    except Exception as e:
        print(f"Error fetching ADA price: {e}")
        return None

def fetch_news_headlines():
    # Static headlines for demo; replace with actual news fetching logic
    return [
        "Cardano Surges After Positive Development Updates",
        "Experts Warn Of Possible Correction in Cardano",
        "Major Financial Institution to Begin Holding ADA Reserves",
        "Bearish Signals Emerge Despite Cardano's Strong Performance",
        "Investors Show Growing Interest in Cardano's Future"
    ]

def analyze_sentiment(headlines):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    scores = []
    for headline in headlines:
        result = sentiment_pipeline(headline)[0]  
        label, score = result["label"], result["score"]
        numeric_score = score if label == "POSITIVE" else -score
        scores.append(numeric_score)
    return np.mean(scores)

def make_decision(avg_sentiment, current_price):
    if avg_sentiment > 0.3:
        return "BUY"
    elif avg_sentiment < -0.3:
        return "SELL"
    else:
        return "HOLD"

def run_agentic_ai():
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
```

---

## Contributing

1. **Fork** the project.  
2. **Create** your feature branch (`git checkout -b feature/my-new-feature`).  
3. **Commit** your changes (`git commit -am 'Add some feature'`).  
4. **Push** to the branch (`git push origin feature/my-new-feature`).  
5. **Open a Pull Request**.

---

## License

This project is licensed under the [MIT License](LICENSE).  

---

## Disclaimer

- **Not Financial Advice**: This is a toy example for educational use only.  
- **No Guarantees**: There is no guarantee of accurate predictions or profitable outcomes. Trading cryptocurrencies involves risk; do your own research.  

---

### Happy experimenting and coding!
