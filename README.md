# Supply Chain Domino Analyzer

> **Predict how supply chain shocks ripple through interconnected companies using Graph Neural Networks**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

![Demo Screenshot](https://via.placeholder.com/800x400/1e293b/ffffff?text=Supply+Chain+Risk+Network+Visualization)

## What It Does

This grade system uses **Graph Neural Networks** to model and predict how supply chain disruptions propagate through interconnected companies. Think of it as "domino effect prediction" for supply chains.

## Quick Start 


1. Download this folder
2. Open `dashboard/enhanced_dashboard.html` in your browser
3. Click any scenario to see the results

That's it! The dashboard works completely offline with built-in risk simulation.

## Key Features

### Interactive Dashboard
- Web interface (React + Tailwind)
- Real-time network visualization
- Monte Carlo simulations (1000+ iterations)
- Portfolio risk analysis

### Graph Neural Network
- 3-layer Graph Attention Network (GAT)
- Learns complex supply chain dependencies
- <100ms prediction time
- Trained on realistic shock scenarios

### Portfolio Analysis
-  Calculate Value-at-Risk (VaR)
- Sector concentration metrics
-  Scenario impact simulations
-  Professional risk reports

###  REST API
-  FastAPI backend
-  RESTful endpoints
- Auto-generated docs (Swagger)
- Real-time predictions

## Usage Examples

### Python API
```python
from src.stress_test import SupplyChainStressTester

# Initialize
tester = SupplyChainStressTester(
    model_path='models/supply_chain_gnn.pt',
    data_path='models/graph_data.pt',
    mappings_path='models/company_mappings.pkl'
)

# Run scenario
results = tester.run_scenario(
    "Taiwan Earthquake",
    affected_companies=['TSMC'],
    intensity=0.9
)
```

### REST API
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "affected_companies": ["TSMC"],
    "intensity": 0.9
  }'
```

### Portfolio Analysis
```python
from src.portfolio_analyzer import PortfolioRiskAnalyzer, PortfolioHolding

holdings = [
    PortfolioHolding('NVDA', 'Nvidia', 100, 400, 875, 'Semiconductors'),
    PortfolioHolding('AAPL', 'Apple', 200, 150, 185, 'Tech'),
]

analyzer = PortfolioRiskAnalyzer(holdings)
report = analyzer.generate_risk_report(risk_scores, "Taiwan Earthquake")
print(report)
```

## Predefined Scenarios

| Scenario | Epicenter | Intensity | Category |
|----------|-----------|-----------|----------|
| Taiwan Earthquake | TSMC | 90% | Natural Disaster |
| Lithium Shortage | Panasonic, LG, CATL | 80% | Resource Constraint |
| AI Chip Constraint | Nvidia | 70% | Demand Surge |
| Rare Earth Ban | ASML, Samsung | 85% | Geopolitical |
| Energy Crisis | TSMC, Intel, Samsung | 75% | Energy |
| Port Strike | Foxconn, Samsung, Sony | 65% | Logistics |

## Tech Stack used

**Backend:** Python, PyTorch, PyTorch Geometric, NetworkX, FastAPI  
**Frontend:** React, Tailwind CSS, Plotly.js, Chart.js  
**Data:** Pandas, NumPy, Scikit-learn  
**Visualization:** Matplotlib, Seaborn, Plotly

## How It Works

### 1. Graph Neural Network
- Represents companies as **nodes** in a graph
- Supply chain relationships as **edges**
- Learns to propagate risk through the network
- Uses attention mechanism to weight connections

### 2. Risk Propagation
```
Direct Impact (90%) → 1st-Order Impact (70%) → 2nd-Order Impact (50%)
  TSMC         →      Apple, Nvidia       →     Microsoft, Tesla
```

### 3. Monte Carlo Simulation
- Runs 1000+ iterations with random variations
- Provides confidence intervals
- Accounts for uncertainty in propagation

##  Performance

- **Training Time:** ~5-10 minutes on CPU
- **Inference Time:** <100ms per scenario
- **Model Size:** ~2MB
- **Accuracy:** MSE < 0.03 on validation set
