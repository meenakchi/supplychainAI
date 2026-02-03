"""
Real-time Market Data Fetcher
Integrates with financial APIs to get current stock prices and news
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import time


class MarketDataFetcher:
    """
    Fetch real-time market data and news for supply chain companies
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data fetcher
        
        Args:
            api_key: API key for financial data service (optional for demo)
        """
        self.api_key = api_key
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def get_stock_price(self, ticker: str) -> Dict:
        """
        Get current stock price and basic metrics
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with price data
        """
        # Check cache
        cache_key = f"price_{ticker}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_data
        
        # In production, use real API like Alpha Vantage, Yahoo Finance, etc.
        # For demo, generate realistic synthetic data
        data = self._generate_synthetic_price(ticker)
        
        self.cache[cache_key] = (time.time(), data)
        return data
    
    def get_multiple_prices(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get prices for multiple stocks
        
        Args:
            tickers: List of ticker symbols
        
        Returns:
            DataFrame with price data
        """
        data = []
        for ticker in tickers:
            price_data = self.get_stock_price(ticker)
            price_data['ticker'] = ticker
            data.append(price_data)
        
        return pd.DataFrame(data)
    
    def get_historical_correlation(self, 
                                   tickers: List[str],
                                   days: int = 90) -> pd.DataFrame:
        """
        Calculate historical price correlation between stocks
        
        Args:
            tickers: List of ticker symbols
            days: Number of days of history
        
        Returns:
            Correlation matrix
        """
        # In production, fetch real historical data
        # For demo, generate synthetic correlation based on supply chain
        correlation_matrix = self._generate_correlation_matrix(tickers)
        return correlation_matrix
    
    def get_supply_chain_news(self, 
                             companies: List[str],
                             days_back: int = 7) -> List[Dict]:
        """
        Fetch recent news about supply chain disruptions
        
        Args:
            companies: List of company names
            days_back: How many days back to search
        
        Returns:
            List of news articles
        """
        # In production, use news API like NewsAPI, Google News, etc.
        # For demo, return synthetic news
        news = self._generate_synthetic_news(companies, days_back)
        return news
    
    def detect_recent_shocks(self, 
                            tickers: List[str],
                            threshold: float = 0.05) -> List[Dict]:
        """
        Detect recent significant price movements that could indicate shocks
        
        Args:
            tickers: List of ticker symbols
            threshold: Price change threshold (e.g., 0.05 = 5%)
        
        Returns:
            List of detected shocks
        """
        shocks = []
        
        for ticker in tickers:
            price_data = self.get_stock_price(ticker)
            
            if abs(price_data['change_percent']) > threshold * 100:
                shocks.append({
                    'ticker': ticker,
                    'company': self._ticker_to_company(ticker),
                    'change': price_data['change_percent'],
                    'timestamp': price_data['timestamp'],
                    'severity': 'high' if abs(price_data['change_percent']) > 10 else 'moderate'
                })
        
        return sorted(shocks, key=lambda x: abs(x['change']), reverse=True)
    
    def get_sector_performance(self) -> Dict[str, float]:
        """
        Get performance by sector
        
        Returns:
            Dictionary of sector -> performance
        """
        sectors = {
            'Semiconductors': np.random.normal(0.5, 2.0),
            'Electric Vehicles': np.random.normal(1.2, 3.0),
            'Consumer Electronics': np.random.normal(-0.3, 1.5),
            'Cloud Computing': np.random.normal(0.8, 2.5),
            'Battery Manufacturing': np.random.normal(0.2, 2.0),
            'Automotive': np.random.normal(-0.5, 1.8)
        }
        return sectors
    
    def _generate_synthetic_price(self, ticker: str) -> Dict:
        """Generate realistic synthetic price data"""
        base_prices = {
            'NVDA': 875.50, 'TSM': 145.30, 'AAPL': 185.20, 'MSFT': 378.90,
            'TSLA': 242.80, 'AMD': 142.50, 'GOOGL': 140.60, 'AMZN': 175.30,
            'INTC': 43.20, 'SSNLF': 1650.00, 'META': 485.20
        }
        
        base_price = base_prices.get(ticker, 100.0)
        
        # Add some random walk
        change = np.random.normal(0, 2.5)
        current_price = base_price * (1 + change / 100)
        
        return {
            'ticker': ticker,
            'price': round(current_price, 2),
            'change': round(current_price - base_price, 2),
            'change_percent': round(change, 2),
            'volume': int(np.random.uniform(1e6, 50e6)),
            'timestamp': datetime.now().isoformat(),
            'market_cap': round(current_price * np.random.uniform(1e9, 3e12), 0)
        }
    
    def _generate_correlation_matrix(self, tickers: List[str]) -> pd.DataFrame:
        """Generate synthetic but realistic correlation matrix"""
        n = len(tickers)
        
        # Start with random correlation
        random_corr = np.random.uniform(0.2, 0.5, (n, n))
        
        # Make it symmetric
        correlation = (random_corr + random_corr.T) / 2
        
        # Set diagonal to 1
        np.fill_diagonal(correlation, 1.0)
        
        # Add some structure based on known relationships
        # Companies in same sector should be more correlated
        sector_groups = {
            'semis': ['NVDA', 'TSM', 'AMD', 'INTC'],
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'META'],
            'ev': ['TSLA']
        }
        
        for sector, group_tickers in sector_groups.items():
            indices = [i for i, t in enumerate(tickers) if t in group_tickers]
            for i in indices:
                for j in indices:
                    if i != j:
                        correlation[i, j] = np.random.uniform(0.6, 0.9)
        
        df = pd.DataFrame(correlation, index=tickers, columns=tickers)
        return df
    
    def _generate_synthetic_news(self, 
                                 companies: List[str],
                                 days_back: int) -> List[Dict]:
        """Generate synthetic news articles"""
        news_templates = [
            "{company} reports supply chain constraints affecting Q{quarter} production",
            "{company} announces new supplier partnership to diversify supply chain",
            "Analysts raise concerns about {company}'s exposure to {region} disruptions",
            "{company} stock falls on semiconductor shortage worries",
            "{company} invests $XB in supply chain resilience initiatives"
        ]
        
        regions = ['Taiwan', 'China', 'Southeast Asia', 'Europe', 'North America']
        
        news = []
        for i in range(min(10, len(companies) * 2)):
            company = np.random.choice(companies)
            template = np.random.choice(news_templates)
            
            article = {
                'title': template.format(
                    company=company,
                    quarter=np.random.choice(['Q1', 'Q2', 'Q3', 'Q4']),
                    region=np.random.choice(regions)
                ),
                'company': company,
                'date': (datetime.now() - timedelta(days=np.random.randint(0, days_back))).isoformat(),
                'sentiment': np.random.choice(['negative', 'neutral', 'positive']),
                'relevance_score': round(np.random.uniform(0.5, 1.0), 2)
            }
            news.append(article)
        
        return sorted(news, key=lambda x: x['date'], reverse=True)
    
    def _ticker_to_company(self, ticker: str) -> str:
        """Map ticker to company name"""
        mapping = {
            'NVDA': 'Nvidia', 'TSM': 'TSMC', 'AAPL': 'Apple',
            'MSFT': 'Microsoft', 'TSLA': 'Tesla', 'AMD': 'AMD',
            'GOOGL': 'Google', 'AMZN': 'Amazon', 'INTC': 'Intel',
            'SSNLF': 'Samsung', 'META': 'Meta'
        }
        return mapping.get(ticker, ticker)


class CompanyTickerMapper:
    """
    Map between company names and ticker symbols
    """
    
    MAPPING = {
        'Nvidia': 'NVDA',
        'TSMC': 'TSM',
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Tesla': 'TSLA',
        'AMD': 'AMD',
        'Google': 'GOOGL',
        'Amazon': 'AMZN',
        'Intel': 'INTC',
        'Samsung': 'SSNLF',
        'Foxconn': '2317.TW',
        'ASML': 'ASML',
        'Panasonic': 'PCRFY',
        'LG': '066570.KS',
        'CATL': '300750.SZ',
        'Ford': 'F',
        'GM': 'GM',
        'Qualcomm': 'QCOM',
        'MediaTek': '2454.TW',
        'Sony': 'SONY',
        'Dell': 'DELL',
        'HP': 'HPQ',
        'Meta': 'META',
        'ByteDance': 'PRIVATE',
        'Alibaba': 'BABA',
        'Tencent': 'TCEHY',
        'SK_Hynix': '000660.KS',
        'Bosch': 'PRIVATE',
        'Denso': 'DNZOY'
    }
    
    @classmethod
    def get_ticker(cls, company: str) -> Optional[str]:
        """Get ticker for company name"""
        return cls.MAPPING.get(company)
    
    @classmethod
    def get_company(cls, ticker: str) -> Optional[str]:
        """Get company name for ticker"""
        reverse = {v: k for k, v in cls.MAPPING.items()}
        return reverse.get(ticker)
    
    @classmethod
    def get_all_tickers(cls) -> List[str]:
        """Get all valid tickers (excluding private companies)"""
        return [t for t in cls.MAPPING.values() if t != 'PRIVATE']


if __name__ == "__main__":
    # Demo
    fetcher = MarketDataFetcher()
    mapper = CompanyTickerMapper()
    
    print("=== Market Data Demo ===\n")
    
    # Get prices
    print("Current Prices:")
    tickers = ['NVDA', 'AAPL', 'TSLA', 'TSMC']
    prices = fetcher.get_multiple_prices(tickers)
    print(prices[['ticker', 'price', 'change_percent']])
    
    # Detect shocks
    print("\n\nRecent Shocks (>5% moves):")
    shocks = fetcher.detect_recent_shocks(mapper.get_all_tickers()[:15])
    for shock in shocks[:5]:
        print(f"  {shock['company']}: {shock['change']:+.2f}% ({shock['severity']})")
    
    # Get news
    print("\n\nRecent Supply Chain News:")
    news = fetcher.get_supply_chain_news(['Nvidia', 'TSMC', 'Apple'], days_back=7)
    for article in news[:3]:
        print(f"  [{article['sentiment'].upper()}] {article['title']}")
    
    # Sector performance
    print("\n\nSector Performance:")
    sectors = fetcher.get_sector_performance()
    for sector, perf in sectors.items():
        print(f"  {sector}: {perf:+.2f}%")