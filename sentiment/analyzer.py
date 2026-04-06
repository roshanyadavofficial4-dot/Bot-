import os
import json
import logging
import asyncio
import requests
from google import genai
from core.api_rotator import GeminiAPIRotator
from bs4 import BeautifulSoup

logger = logging.getLogger("SentimentAnalyzer")

class SentimentAnalyzer:
    """
    NLP-based Sentiment Engine for Project Digital Evolution.
    Utilizes Gemini Pro to analyze real-time news and community fear/greed metrics,
    providing a normalized sentiment score and Market Crash Protection.
    """
    
    def __init__(self):
        # Initialize the API Key Rotator (Error 1 Fix)
        try:
            self.rotator = GeminiAPIRotator()
            self.ai_clients = {} # Cache for configured models per key
        except Exception as e:
            self.rotator = None
            logger.warning(f"Failed to initialize GeminiAPIRotator: {e}. Sentiment analysis will be limited.")
            
        # Using a valid Gemini model name
        self.model_name = 'gemini-1.5-pro-latest'

    def _get_ai_client(self, api_key: str):
        """Returns a cached or new Client instance for the given key."""
        if api_key not in self.ai_clients:
            self.ai_clients[api_key] = genai.Client(api_key=api_key)
        return self.ai_clients[api_key]

    async def fetch_fear_and_greed(self) -> float:
        """
        Fetches the Crypto Fear & Greed Index from alternative.me.
        Returns a normalized score between -1.0 (Extreme Fear) and 1.0 (Extreme Greed).
        """
        try:
            # Run synchronous request in a thread to avoid blocking the event loop
            response = await asyncio.to_thread(requests.get, "https://api.alternative.me/fng/?limit=1", timeout=5)
            data = response.json()
            fng_value = float(data['data'][0]['value'])
            
            # Normalize from [0, 100] to [-1.0, 1.0]
            # 50 becomes 0.0, 0 becomes -1.0, 100 becomes 1.0
            normalized_score = (fng_value / 50.0) - 1.0
            return normalized_score
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed index: {e}")
            return 0.0

    async def scrape_news(self, symbol: str) -> list:
        """
        Scrapes real-time financial news headlines related to the specific crypto symbol.
        Uses CryptoCompare's free news API as a highly reliable proxy for crypto news feeds.
        """
        base_coin = symbol.split('/')[0] if '/' in symbol else symbol
        headlines = []
        
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            response = await asyncio.to_thread(requests.get, url, timeout=5)
            data = response.json()
            
            if data.get("Data"):
                for article in data["Data"]:
                    # Filter for articles mentioning the specific coin in title or categories
                    if base_coin.upper() in article['categories'].upper() or base_coin.upper() in article['title'].upper():
                        headlines.append(article['title'])
                        if len(headlines) >= 15:  # Limit to top 15 most recent relevant headlines
                            break
                            
        except Exception as e:
            logger.error(f"Error scraping news for {symbol}: {e}")
            
        return headlines

    async def analyze_sentiment(self, headlines: list) -> float:
        """
        Sends scraped headlines to Gemini Pro to get a normalized sentiment score (-1.0 to +1.0).
        """
        if not headlines or not self.rotator:
            return 0.0

        active_key = self.rotator.get_active_key()
        ai_client = self._get_ai_client(active_key)

        prompt = f"""
        Act as an expert quantitative financial analyst. 
        Analyze the following recent news headlines for the cryptocurrency market.
        Determine the overall sentiment on a scale from -1.0 (Extreme Panic/Bearish) to +1.0 (Extreme Euphoria/Bullish).
        0.0 represents completely neutral.
        
        Headlines:
        {json.dumps(headlines, indent=2)}
        
        Respond ONLY with a single float number representing the score. Do not include any other text, markdown, or explanation.
        """
        
        try:
            # Re-configure for this specific call to ensure the right key is used
            response = await asyncio.to_thread(
                ai_client.models.generate_content,
                model=self.model_name,
                contents=prompt
            )
            
            score_text = response.text.strip()
            score = float(score_text)
            
            # Clamp the score strictly between -1.0 and 1.0
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                self.rotator.flag_rate_limit(active_key)
            logger.error(f"Gemini API error during sentiment analysis: {e}")
            return 0.0

    async def get_sentiment_mood(self, symbol: str) -> dict:
        """
        Fast, non-blocking function to return the overall 'Sentiment Mood'.
        Combines Fear & Greed index with Gemini-analyzed news sentiment.
        """
        # 1. Fetch F&G and News concurrently for maximum speed
        fng_task = asyncio.create_task(self.fetch_fear_and_greed())
        news_task = asyncio.create_task(self.scrape_news(symbol))
        
        fng_score, headlines = await asyncio.gather(fng_task, news_task)
        
        # 2. Analyze news sentiment via Gemini
        news_score = await self.analyze_sentiment(headlines) if headlines else 0.0
        
        # 3. Calculate Weighted Average
        # News is specific to the coin (70% weight), F&G is macro market sentiment (30% weight)
        if headlines:
            final_score = (news_score * 0.7) + (fng_score * 0.3)
        else:
            final_score = fng_score
            
        final_score = round(final_score, 4)
        
        # 4. Market Crash Protection Logic
        signal = "NEUTRAL"
        if final_score <= -0.6:
            signal = "CRASH_WARNING"  # Triggers bot pause
        elif final_score >= 0.6:
            signal = "EUPHORIA_WARNING" # Triggers tight trailing stops
        elif final_score > 0.2:
            signal = "BULLISH"
        elif final_score < -0.2:
            signal = "BEARISH"
            
        logger.info(
            f"Sentiment [{symbol}] | Final: {final_score} | Signal: {signal} | "
            f"News: {news_score:.2f} | F&G: {fng_score:.2f}"
        )
        
        return {
            "symbol": symbol,
            "score": final_score,
            "signal": signal,
            "fng_score": fng_score,
            "news_score": news_score,
            "headlines_analyzed": len(headlines)
        }

# Example usage (if run directly):
# async def main():
#     analyzer = SentimentAnalyzer()
#     result = await analyzer.get_sentiment_mood("BTC/USDT")
#     print(result)
#
# if __name__ == "__main__":
#     asyncio.run(main())
