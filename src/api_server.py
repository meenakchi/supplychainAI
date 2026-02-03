"""
FastAPI Backend for Supply Chain Risk Analyzer
Provides REST API for risk predictions and scenario analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import pickle
import uvicorn
from datetime import datetime

# Import our modules
import sys
sys.path.append('/home/claude/src')
from model import SupplyChainGNN, CrisisScenarios
from data_fetcher import MarketDataFetcher, CompanyTickerMapper


app = FastAPI(
    title="Supply Chain Risk Analyzer API",
    description="Predict supply chain risk propagation using Graph Neural Networks",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class ShockScenario(BaseModel):
    """Request model for shock scenario"""
    affected_companies: List[str] = Field(..., description="Companies experiencing the shock")
    intensity: float = Field(0.8, ge=0.0, le=1.0, description="Shock intensity (0-1)")
    scenario_name: Optional[str] = Field(None, description="Optional scenario name")


class RiskPrediction(BaseModel):
    """Response model for risk prediction"""
    company: str
    risk_score: float
    risk_level: str
    ticker: Optional[str]
    is_epicenter: bool


class ScenarioResult(BaseModel):
    """Complete scenario analysis result"""
    scenario_name: str
    affected_companies: List[str]
    intensity: float
    timestamp: str
    predictions: List[RiskPrediction]
    summary: Dict[str, int]


class CompanyInfo(BaseModel):
    """Company information"""
    name: str
    ticker: Optional[str]
    suppliers: List[str]
    customers: List[str]
    centrality_metrics: Dict[str, float]


# Global state
model_state = {
    'model': None,
    'data': None,
    'mappings': None,
    'fetcher': None,
    'device': None
}


@app.on_event("startup")
async def load_model():
    """Load model and data on startup"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load graph data
        data = torch.load('/home/claude/models/graph_data.pt', map_location=device)
        
        # Load mappings
        with open('/home/claude/models/company_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        # Initialize model
        model = SupplyChainGNN(
            input_dim=data.x.shape[1] + 1,
            hidden_dim=64,
            num_layers=3,
            dropout=0.2,
            heads=4
        )
        
        # Load weights
        checkpoint = torch.load('/home/claude/models/supply_chain_gnn.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Initialize market data fetcher
        fetcher = MarketDataFetcher()
        
        # Store in global state
        model_state['model'] = model
        model_state['data'] = data
        model_state['mappings'] = mappings
        model_state['fetcher'] = fetcher
        model_state['device'] = device
        
        print("✓ Model loaded successfully")
        print(f"✓ Running on {device}")
        print(f"✓ Tracking {len(mappings['company_to_idx'])} companies")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        # Continue with demo mode


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Supply Chain Risk Analyzer API",
        "status": "operational" if model_state['model'] is not None else "demo_mode",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/predict",
            "scenarios": "/api/scenarios",
            "companies": "/api/companies",
            "market": "/api/market"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_state['model'] is not None,
        "device": str(model_state['device']) if model_state['device'] else "none",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/predict", response_model=ScenarioResult)
async def predict_risk(scenario: ShockScenario):
    """
    Predict supply chain risk for a shock scenario
    
    Args:
        scenario: Shock scenario details
    
    Returns:
        Risk predictions for all companies
    """
    if model_state['model'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract model components
        model = model_state['model']
        data = model_state['data']
        company_to_idx = model_state['mappings']['company_to_idx']
        idx_to_company = model_state['mappings']['idx_to_company']
        device = model_state['device']
        
        # Prepare features
        x = data.x.clone().to(device)
        edge_index = data.edge_index.to(device)
        
        # Add shock feature
        shock_feature = torch.zeros(x.shape[0], 1).to(device)
        for company in scenario.affected_companies:
            if company in company_to_idx:
                idx = company_to_idx[company]
                shock_feature[idx] = scenario.intensity
        
        # Augment features
        x_augmented = torch.cat([x, shock_feature], dim=1)
        
        # Predict
        with torch.no_grad():
            risk_scores = model(x_augmented, edge_index)
        
        # Build response
        predictions = []
        risk_counts = {'critical': 0, 'high': 0, 'moderate': 0, 'low': 0}
        
        for idx, company in idx_to_company.items():
            risk = float(risk_scores[idx].cpu().item())
            
            # Categorize risk
            if risk > 0.7:
                level = 'critical'
                risk_counts['critical'] += 1
            elif risk > 0.5:
                level = 'high'
                risk_counts['high'] += 1
            elif risk > 0.3:
                level = 'moderate'
                risk_counts['moderate'] += 1
            else:
                level = 'low'
                risk_counts['low'] += 1
            
            predictions.append(RiskPrediction(
                company=company,
                risk_score=round(risk, 4),
                risk_level=level,
                ticker=CompanyTickerMapper.get_ticker(company),
                is_epicenter=company in scenario.affected_companies
            ))
        
        # Sort by risk score
        predictions.sort(key=lambda x: x.risk_score, reverse=True)
        
        return ScenarioResult(
            scenario_name=scenario.scenario_name or "Custom Scenario",
            affected_companies=scenario.affected_companies,
            intensity=scenario.intensity,
            timestamp=datetime.now().isoformat(),
            predictions=predictions,
            summary=risk_counts
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/scenarios")
async def list_scenarios():
    """List all predefined crisis scenarios"""
    scenarios = [
        CrisisScenarios.taiwan_earthquake(),
        CrisisScenarios.lithium_shortage(),
        CrisisScenarios.nvidia_supply_constraint(),
        CrisisScenarios.china_export_restrictions(),
        CrisisScenarios.energy_crisis()
    ]
    
    return {
        "scenarios": scenarios,
        "count": len(scenarios)
    }


@app.get("/api/companies")
async def list_companies():
    """List all companies in the network"""
    if model_state['mappings'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    idx_to_company = model_state['mappings']['idx_to_company']
    
    companies = []
    for company in idx_to_company.values():
        companies.append({
            'name': company,
            'ticker': CompanyTickerMapper.get_ticker(company)
        })
    
    return {
        "companies": sorted(companies, key=lambda x: x['name']),
        "count": len(companies)
    }


@app.get("/api/companies/{company_name}", response_model=CompanyInfo)
async def get_company_info(company_name: str):
    """
    Get detailed information about a specific company
    
    Args:
        company_name: Name of the company
    
    Returns:
        Company information including suppliers and customers
    """
    if model_state['data'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # In production, load the graph and compute neighbors
    # For now, return mock data
    return CompanyInfo(
        name=company_name,
        ticker=CompanyTickerMapper.get_ticker(company_name),
        suppliers=['Supplier1', 'Supplier2'],
        customers=['Customer1', 'Customer2'],
        centrality_metrics={
            'pagerank': 0.05,
            'betweenness': 0.12,
            'degree': 8
        }
    )


@app.get("/api/market/prices")
async def get_market_prices(tickers: Optional[str] = None):
    """
    Get current market prices for companies
    
    Args:
        tickers: Comma-separated list of tickers (optional)
    
    Returns:
        Current price data
    """
    fetcher = model_state.get('fetcher') or MarketDataFetcher()
    
    if tickers:
        ticker_list = tickers.split(',')
    else:
        ticker_list = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'AMD']
    
    prices = fetcher.get_multiple_prices(ticker_list)
    return prices.to_dict(orient='records')


@app.get("/api/market/shocks")
async def detect_market_shocks(threshold: float = 0.05):
    """
    Detect recent significant market movements
    
    Args:
        threshold: Minimum change threshold (default 5%)
    
    Returns:
        List of detected shocks
    """
    fetcher = model_state.get('fetcher') or MarketDataFetcher()
    
    all_tickers = CompanyTickerMapper.get_all_tickers()[:20]  # Limit for performance
    shocks = fetcher.detect_recent_shocks(all_tickers, threshold)
    
    return {
        "shocks": shocks,
        "count": len(shocks),
        "threshold": threshold,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/market/news")
async def get_supply_chain_news(days: int = 7):
    """
    Get recent supply chain news
    
    Args:
        days: Number of days to look back
    
    Returns:
        Recent news articles
    """
    fetcher = model_state.get('fetcher') or MarketDataFetcher()
    
    companies = ['Nvidia', 'TSMC', 'Apple', 'Tesla', 'Samsung']
    news = fetcher.get_supply_chain_news(companies, days)
    
    return {
        "news": news,
        "count": len(news),
        "days_back": days
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )