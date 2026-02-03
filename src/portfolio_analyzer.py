"""
Portfolio Risk Analyzer
Analyzes how supply chain shocks affect actual investment portfolios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PortfolioHolding:
    """Represents a single portfolio holding"""
    ticker: str
    company: str
    shares: float
    cost_basis: float
    current_price: float
    sector: str
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_gain_loss(self) -> float:
        return self.market_value - (self.shares * self.cost_basis)
    
    @property
    def return_pct(self) -> float:
        return (self.current_price - self.cost_basis) / self.cost_basis * 100


class PortfolioRiskAnalyzer:
    """
    Analyze portfolio exposure to supply chain risks
    """
    
    def __init__(self, holdings: List[PortfolioHolding]):
        """
        Initialize portfolio analyzer
        
        Args:
            holdings: List of portfolio holdings
        """
        self.holdings = holdings
        self.total_value = sum(h.market_value for h in holdings)
        
    def calculate_exposure_by_sector(self) -> pd.DataFrame:
        """
        Calculate portfolio exposure by sector
        
        Returns:
            DataFrame with sector exposure
        """
        sector_data = {}
        
        for holding in self.holdings:
            if holding.sector not in sector_data:
                sector_data[holding.sector] = {
                    'value': 0,
                    'companies': []
                }
            
            sector_data[holding.sector]['value'] += holding.market_value
            sector_data[holding.sector]['companies'].append(holding.company)
        
        data = []
        for sector, info in sector_data.items():
            data.append({
                'Sector': sector,
                'Market_Value': info['value'],
                'Percentage': info['value'] / self.total_value * 100,
                'Companies': len(info['companies']),
                'Company_List': ', '.join(info['companies'])
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Market_Value', ascending=False)
        return df
    
    def calculate_supply_chain_exposure(self,
                                       company_risk_scores: Dict[str, float]) -> pd.DataFrame:
        """
        Calculate portfolio risk based on supply chain vulnerability
        
        Args:
            company_risk_scores: Dictionary of company -> risk score
        
        Returns:
            DataFrame with risk analysis
        """
        data = []
        
        for holding in self.holdings:
            risk_score = company_risk_scores.get(holding.company, 0.0)
            
            # Calculate value at risk
            var_pessimistic = holding.market_value * risk_score * 0.5  # Assume 50% max impact
            var_realistic = holding.market_value * risk_score * 0.25  # Assume 25% realistic impact
            
            data.append({
                'Company': holding.company,
                'Ticker': holding.ticker,
                'Sector': holding.sector,
                'Market_Value': holding.market_value,
                'Portfolio_Weight': holding.market_value / self.total_value * 100,
                'Risk_Score': risk_score,
                'Risk_Level': self._categorize_risk(risk_score),
                'VaR_Pessimistic': var_pessimistic,
                'VaR_Realistic': var_realistic,
                'Current_Return': holding.return_pct
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Risk_Score', ascending=False)
        return df
    
    def simulate_shock_impact(self,
                            company_risk_scores: Dict[str, float],
                            market_correlation: float = 0.3) -> Dict:
        """
        Simulate the impact of a supply chain shock on portfolio value
        
        Args:
            company_risk_scores: Risk scores for each company
            market_correlation: Overall market correlation factor
        
        Returns:
            Dictionary with simulation results
        """
        scenarios = {
            'pessimistic': {'direct': 0.5, 'indirect': 0.15},  # Direct 50% loss, indirect 15%
            'realistic': {'direct': 0.25, 'indirect': 0.08},   # Direct 25% loss, indirect 8%
            'optimistic': {'direct': 0.10, 'indirect': 0.03}   # Direct 10% loss, indirect 3%
        }
        
        results = {}
        
        for scenario_name, factors in scenarios.items():
            total_loss = 0
            
            for holding in self.holdings:
                risk = company_risk_scores.get(holding.company, 0.0)
                
                # Direct impact on highly exposed companies
                if risk > 0.5:
                    direct_loss = holding.market_value * risk * factors['direct']
                    total_loss += direct_loss
                
                # Indirect impact on all holdings due to market correlation
                indirect_loss = holding.market_value * factors['indirect'] * market_correlation
                total_loss += indirect_loss
            
            loss_pct = total_loss / self.total_value * 100
            
            results[scenario_name] = {
                'total_loss': total_loss,
                'loss_percentage': loss_pct,
                'remaining_value': self.total_value - total_loss,
                'impact_level': self._categorize_impact(loss_pct)
            }
        
        return results
    
    def calculate_concentration_risk(self) -> Dict:
        """
        Calculate portfolio concentration risk
        
        Returns:
            Dictionary with concentration metrics
        """
        # Herfindahl-Hirschman Index (HHI) - measures concentration
        weights = np.array([h.market_value / self.total_value for h in self.holdings])
        hhi = np.sum(weights ** 2)
        
        # Top holdings concentration
        sorted_holdings = sorted(self.holdings, key=lambda h: h.market_value, reverse=True)
        top_5_value = sum(h.market_value for h in sorted_holdings[:5])
        top_5_pct = top_5_value / self.total_value * 100
        
        # Sector concentration
        sector_exposure = self.calculate_exposure_by_sector()
        max_sector_exposure = sector_exposure['Percentage'].max()
        
        return {
            'herfindahl_index': hhi,
            'concentration_level': self._interpret_hhi(hhi),
            'top_5_concentration': top_5_pct,
            'max_sector_exposure': max_sector_exposure,
            'number_of_holdings': len(self.holdings),
            'diversification_score': self._calculate_diversification_score(hhi, len(self.holdings))
        }
    
    def generate_risk_report(self,
                           company_risk_scores: Dict[str, float],
                           scenario_name: str = "Supply Chain Shock") -> str:
        """
        Generate a comprehensive risk report
        
        Args:
            company_risk_scores: Risk scores from GNN model
            scenario_name: Name of the scenario
        
        Returns:
            Formatted risk report
        """
        report = []
        report.append("=" * 80)
        report.append(f"PORTFOLIO RISK ANALYSIS: {scenario_name}")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Portfolio Value: ${self.total_value:,.2f}")
        report.append(f"Number of Holdings: {len(self.holdings)}")
        report.append("")
        
        # Exposure analysis
        exposure = self.calculate_supply_chain_exposure(company_risk_scores)
        
        report.append("TOP 5 MOST VULNERABLE HOLDINGS:")
        report.append("-" * 80)
        for idx, row in exposure.head(5).iterrows():
            report.append(
                f"{row['Company']:<20} "
                f"Risk: {row['Risk_Score']:>6.1%}  "
                f"Value: ${row['Market_Value']:>12,.2f}  "
                f"Weight: {row['Portfolio_Weight']:>5.1f}%  "
                f"VaR: ${row['VaR_Realistic']:>10,.2f}"
            )
        report.append("")
        
        # Concentration risk
        concentration = self.calculate_concentration_risk()
        report.append("CONCENTRATION RISK:")
        report.append("-" * 80)
        report.append(f"Diversification Score: {concentration['diversification_score']:.2f}/10")
        report.append(f"Top 5 Holdings: {concentration['top_5_concentration']:.1f}% of portfolio")
        report.append(f"Largest Sector Exposure: {concentration['max_sector_exposure']:.1f}%")
        report.append("")
        
        # Scenario impact
        impact = self.simulate_shock_impact(company_risk_scores)
        report.append("SCENARIO IMPACT ANALYSIS:")
        report.append("-" * 80)
        for scenario, results in impact.items():
            report.append(
                f"{scenario.upper():<15} "
                f"Loss: ${results['total_loss']:>12,.2f} ({results['loss_percentage']:>5.2f}%)  "
                f"Impact: {results['impact_level']}"
            )
        report.append("")
        
        # Sector exposure
        sector_exp = self.calculate_exposure_by_sector()
        report.append("SECTOR EXPOSURE:")
        report.append("-" * 80)
        for idx, row in sector_exp.iterrows():
            report.append(
                f"{row['Sector']:<25} "
                f"{row['Percentage']:>6.1f}%  "
                f"${row['Market_Value']:>12,.2f}  "
                f"({row['Companies']} companies)"
            )
        report.append("")
        
        # Recommendations
        high_risk_holdings = len(exposure[exposure['Risk_Score'] > 0.5])
        report.append("RECOMMENDATIONS:")
        report.append("-" * 80)
        
        if high_risk_holdings > 0:
            report.append(f"⚠️  {high_risk_holdings} holdings have HIGH supply chain risk (>50%)")
            report.append("   Consider: hedging positions, reducing exposure, or diversifying")
        
        if concentration['top_5_concentration'] > 50:
            report.append("⚠️  Portfolio is highly concentrated in top 5 holdings")
            report.append("   Consider: adding more positions to improve diversification")
        
        if concentration['max_sector_exposure'] > 40:
            report.append("⚠️  Significant concentration in one sector")
            report.append("   Consider: diversifying across multiple sectors")
        
        if not any([high_risk_holdings > 0, 
                   concentration['top_5_concentration'] > 50,
                   concentration['max_sector_exposure'] > 40]):
            report.append("✓  Portfolio appears well-diversified with manageable supply chain risk")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def visualize_risk_exposure(self,
                               company_risk_scores: Dict[str, float]) -> plt.Figure:
        """
        Create visualization of portfolio risk exposure
        
        Args:
            company_risk_scores: Risk scores from model
        
        Returns:
            Matplotlib figure
        """
        exposure = self.calculate_supply_chain_exposure(company_risk_scores)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Risk Score vs Portfolio Weight (bubble chart)
        ax = axes[0, 0]
        scatter = ax.scatter(
            exposure['Portfolio_Weight'],
            exposure['Risk_Score'] * 100,
            s=exposure['Market_Value'] / 1000,
            alpha=0.6,
            c=exposure['Risk_Score'],
            cmap='RdYlGn_r'
        )
        
        for idx, row in exposure.iterrows():
            if row['Risk_Score'] > 0.4 or row['Portfolio_Weight'] > 10:
                ax.annotate(
                    row['Company'],
                    (row['Portfolio_Weight'], row['Risk_Score'] * 100),
                    fontsize=8,
                    alpha=0.7
                )
        
        ax.set_xlabel('Portfolio Weight (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Risk Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Portfolio Holdings: Risk vs Weight', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Risk Score')
        
        # 2. Value at Risk by Holding
        ax = axes[0, 1]
        top_risk = exposure.head(10)
        y_pos = np.arange(len(top_risk))
        
        bars = ax.barh(y_pos, top_risk['VaR_Realistic'], alpha=0.7)
        
        # Color bars by risk level
        colors = []
        for score in top_risk['Risk_Score']:
            if score > 0.7:
                colors.append('#FF4444')
            elif score > 0.5:
                colors.append('#FF9944')
            elif score > 0.3:
                colors.append('#FFDD44')
            else:
                colors.append('#44DD44')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_risk['Company'])
        ax.set_xlabel('Value at Risk ($)', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Holdings by VaR (Realistic)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 3. Sector Exposure
        ax = axes[1, 0]
        sector_exp = self.calculate_exposure_by_sector()
        
        colors_sector = plt.cm.Set3(np.linspace(0, 1, len(sector_exp)))
        wedges, texts, autotexts = ax.pie(
            sector_exp['Percentage'],
            labels=sector_exp['Sector'],
            autopct='%1.1f%%',
            colors=colors_sector,
            startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Portfolio Allocation by Sector', fontsize=14, fontweight='bold')
        
        # 4. Scenario Impact Comparison
        ax = axes[1, 1]
        impact = self.simulate_shock_impact(company_risk_scores)
        
        scenarios = list(impact.keys())
        losses = [impact[s]['loss_percentage'] for s in scenarios]
        
        bars = ax.bar(scenarios, losses, alpha=0.7, color=['#FF4444', '#FF9944', '#44DD44'])
        ax.set_ylabel('Portfolio Loss (%)', fontsize=12, fontweight='bold')
        ax.set_title('Scenario Impact Analysis', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for i, (scenario, loss) in enumerate(zip(scenarios, losses)):
            ax.text(i, loss + 0.5, f'{loss:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _categorize_risk(score: float) -> str:
        """Categorize risk score"""
        if score > 0.7:
            return "Critical"
        elif score > 0.5:
            return "High"
        elif score > 0.3:
            return "Moderate"
        else:
            return "Low"
    
    @staticmethod
    def _categorize_impact(loss_pct: float) -> str:
        """Categorize impact level"""
        if loss_pct > 15:
            return "Severe"
        elif loss_pct > 10:
            return "Major"
        elif loss_pct > 5:
            return "Moderate"
        else:
            return "Minor"
    
    @staticmethod
    def _interpret_hhi(hhi: float) -> str:
        """Interpret Herfindahl-Hirschman Index"""
        if hhi > 0.25:
            return "Highly Concentrated"
        elif hhi > 0.15:
            return "Moderately Concentrated"
        else:
            return "Well Diversified"
    
    @staticmethod
    def _calculate_diversification_score(hhi: float, num_holdings: int) -> float:
        """
        Calculate a diversification score from 0-10
        
        Args:
            hhi: Herfindahl-Hirschman Index
            num_holdings: Number of holdings
        
        Returns:
            Score from 0 (poor) to 10 (excellent)
        """
        # Ideal HHI for well-diversified portfolio is around 0.05-0.10
        # Penalize both high concentration and too few holdings
        
        hhi_score = max(0, 10 - (hhi - 0.05) * 50)  # Penalty for high HHI
        holdings_score = min(10, num_holdings / 2)  # Benefit for more holdings (cap at 20)
        
        return (hhi_score + holdings_score) / 2


# Demo usage
if __name__ == "__main__":
    # Create sample portfolio
    holdings = [
        PortfolioHolding('NVDA', 'Nvidia', 100, 400, 875, 'Semiconductors'),
        PortfolioHolding('AAPL', 'Apple', 200, 150, 185, 'Consumer Electronics'),
        PortfolioHolding('TSLA', 'Tesla', 50, 200, 242, 'Electric Vehicles'),
        PortfolioHolding('MSFT', 'Microsoft', 100, 300, 378, 'Cloud Computing'),
        PortfolioHolding('TSM', 'TSMC', 150, 100, 145, 'Semiconductors'),
        PortfolioHolding('AMD', 'AMD', 75, 120, 142, 'Semiconductors'),
    ]
    
    analyzer = PortfolioRiskAnalyzer(holdings)
    
    # Sample risk scores (would come from GNN model)
    risk_scores = {
        'Nvidia': 0.65,
        'Apple': 0.72,
        'Tesla': 0.58,
        'Microsoft': 0.45,
        'TSMC': 0.90,
        'AMD': 0.68
    }
    
    # Generate report
    report = analyzer.generate_risk_report(risk_scores, "Taiwan Earthquake Scenario")
    print(report)
    
    # Create visualization
    fig = analyzer.visualize_risk_exposure(risk_scores)
    plt.savefig('portfolio_risk_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to portfolio_risk_analysis.png")