"""
Interactive Stress Testing Interface
Run crisis scenarios and analyze supply chain vulnerabilities
"""

import torch
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import SupplyChainDataLoader
from model import SupplyChainGNN, StressTestEngine, CrisisScenarios
from graph_builder import SupplyChainGraphAnalyzer


class SupplyChainStressTester:
    """
    Complete stress testing system
    """
    
    def __init__(self, model_path: str, data_path: str, mappings_path: str):
        """
        Initialize the stress tester
        
        Args:
            model_path: Path to trained model
            data_path: Path to graph data
            mappings_path: Path to company mappings
        """
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = torch.load(data_path, map_location=self.device)
        
        # Load mappings
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        self.company_to_idx = mappings['company_to_idx']
        self.idx_to_company = mappings['idx_to_company']
        
        # Initialize model
        self.model = SupplyChainGNN(
            input_dim=self.data.x.shape[1] + 1,  # +1 for shock feature
            hidden_dim=64,
            num_layers=3,
            dropout=0.2,
            heads=4
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Running on {self.device}")
        print(f"✓ Tracking {len(self.company_to_idx)} companies")
    
    def predict_impact(self, 
                      shock_companies: List[str],
                      shock_intensity: float = 0.8) -> Dict[str, float]:
        """
        Predict impact of a shock scenario
        
        Args:
            shock_companies: Companies experiencing shock
            shock_intensity: Shock intensity (0-1)
        
        Returns:
            Dictionary of company -> risk score
        """
        # Prepare features
        x = self.data.x.clone().to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        
        # Add shock feature
        shock_feature = torch.zeros(x.shape[0], 1).to(self.device)
        for company in shock_companies:
            if company in self.company_to_idx:
                idx = self.company_to_idx[company]
                shock_feature[idx] = shock_intensity
        
        # Augment features
        x_augmented = torch.cat([x, shock_feature], dim=1)
        
        # Predict
        with torch.no_grad():
            risk_scores = self.model(x_augmented, edge_index)
        
        # Convert to dictionary
        results = {}
        for idx, company in self.idx_to_company.items():
            results[company] = float(risk_scores[idx].cpu().item())
        
        return results
    
    def run_scenario(self, 
                    scenario_name: str,
                    affected_companies: List[str],
                    intensity: float = 0.8,
                    top_k: int = 15) -> pd.DataFrame:
        """
        Run a complete stress test scenario
        
        Args:
            scenario_name: Name of the scenario
            affected_companies: Directly affected companies
            intensity: Shock intensity
            top_k: Number of top results to return
        
        Returns:
            DataFrame with results
        """
        print("\n" + "="*70)
        print(f"🚨 STRESS TEST: {scenario_name}")
        print("="*70)
        print(f"📍 Epicenter: {', '.join(affected_companies)}")
        print(f"⚡ Intensity: {intensity:.0%}")
        print()
        
        # Run prediction
        risk_scores = self.predict_impact(affected_companies, intensity)
        
        # Create results dataframe
        results = []
        for company, risk in risk_scores.items():
            results.append({
                'Company': company,
                'Risk_Score': risk,
                'Risk_Level': self._categorize_risk(risk),
                'Is_Epicenter': company in affected_companies
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('Risk_Score', ascending=False)
        
        # Print summary
        high_risk = len(df[df['Risk_Score'] > 0.5])
        moderate_risk = len(df[(df['Risk_Score'] > 0.3) & (df['Risk_Score'] <= 0.5)])
        
        print(f"📊 IMPACT SUMMARY")
        print(f"   🔴 High Risk (>50%):     {high_risk} companies")
        print(f"   🟡 Moderate Risk (30-50%): {moderate_risk} companies")
        print(f"\n💥 TOP {top_k} MOST VULNERABLE COMPANIES:")
        print()
        
        # Pretty print top results
        for idx, row in df.head(top_k).iterrows():
            emoji = self._get_risk_emoji(row['Risk_Score'])
            epicenter = " ⚠️ [EPICENTER]" if row['Is_Epicenter'] else ""
            print(f"   {emoji} {row['Company']:<20} {row['Risk_Score']:>6.1%}{epicenter}")
        
        print("\n" + "="*70 + "\n")
        
        return df
    
    def compare_scenarios(self, scenarios: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple scenarios side by side
        
        Args:
            scenarios: List of scenario dictionaries
        
        Returns:
            Comparison dataframe
        """
        all_results = {}
        
        for scenario in scenarios:
            risk_scores = self.predict_impact(
                scenario['affected'],
                scenario.get('intensity', 0.8)
            )
            all_results[scenario['name']] = risk_scores
        
        # Create comparison dataframe
        df = pd.DataFrame(all_results)
        df['Company'] = df.index
        df = df[['Company'] + [s['name'] for s in scenarios]]
        
        # Calculate average risk across scenarios
        scenario_cols = [s['name'] for s in scenarios]
        df['Average_Risk'] = df[scenario_cols].mean(axis=1)
        df = df.sort_values('Average_Risk', ascending=False)
        
        return df
    
    def visualize_impact(self, 
                        scenario_name: str,
                        risk_scores: Dict[str, float],
                        top_k: int = 20) -> plt.Figure:
        """
        Visualize scenario impact
        
        Args:
            scenario_name: Scenario name
            risk_scores: Dictionary of risk scores
            top_k: Number of companies to show
        
        Returns:
            Matplotlib figure
        """
        # Sort and get top K
        sorted_companies = sorted(risk_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:top_k]
        
        companies = [c for c, _ in sorted_companies]
        scores = [s for _, s in sorted_companies]
        
        # Create color map
        colors = []
        for score in scores:
            if score > 0.7:
                colors.append('#FF4444')  # Red
            elif score > 0.5:
                colors.append('#FF9944')  # Orange
            elif score > 0.3:
                colors.append('#FFDD44')  # Yellow
            else:
                colors.append('#44DD44')  # Green
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(companies, scores, color=colors, alpha=0.8)
        
        ax.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Supply Chain Impact: {scenario_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.02, i, f'{score:.1%}', 
                   va='center', fontsize=9, fontweight='bold')
        
        # Add risk level legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF4444', alpha=0.8, label='Critical (>70%)'),
            Patch(facecolor='#FF9944', alpha=0.8, label='High (50-70%)'),
            Patch(facecolor='#FFDD44', alpha=0.8, label='Moderate (30-50%)'),
            Patch(facecolor='#44DD44', alpha=0.8, label='Low (<30%)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, 
                       scenarios: List[Dict],
                       output_path: str = 'stress_test_report.txt'):
        """
        Generate a comprehensive stress test report
        
        Args:
            scenarios: List of scenarios to test
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUPPLY CHAIN STRESS TEST REPORT\n")
            f.write("="*80 + "\n\n")
            
            for scenario in scenarios:
                f.write(f"\nSCENARIO: {scenario['name']}\n")
                f.write("-"*80 + "\n")
                f.write(f"Description: {scenario.get('description', 'N/A')}\n")
                f.write(f"Affected: {', '.join(scenario['affected'])}\n")
                f.write(f"Intensity: {scenario.get('intensity', 0.8):.0%}\n\n")
                
                # Run scenario
                risk_scores = self.predict_impact(
                    scenario['affected'],
                    scenario.get('intensity', 0.8)
                )
                
                # Sort and categorize
                sorted_scores = sorted(risk_scores.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)
                
                critical = [c for c, s in sorted_scores if s > 0.7]
                high = [c for c, s in sorted_scores if 0.5 < s <= 0.7]
                moderate = [c for c, s in sorted_scores if 0.3 < s <= 0.5]
                
                f.write(f"Critical Risk ({len(critical)}): {', '.join(critical)}\n\n")
                f.write(f"High Risk ({len(high)}): {', '.join(high)}\n\n")
                f.write(f"Moderate Risk ({len(moderate)}): {', '.join(moderate)}\n\n")
                
                f.write("Top 10 Most Vulnerable:\n")
                for i, (company, score) in enumerate(sorted_scores[:10], 1):
                    f.write(f"  {i}. {company:<20} {score:>6.1%}\n")
                
                f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Report saved to {output_path}")
    
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
    def _get_risk_emoji(score: float) -> str:
        """Get emoji for risk level"""
        if score > 0.7:
            return "🔴"
        elif score > 0.5:
            return "🟠"
        elif score > 0.3:
            return "🟡"
        else:
            return "🟢"


def main():
    """Demo of stress testing system"""
    
    # Initialize stress tester
    tester = SupplyChainStressTester(
        model_path='models/supply_chain_gnn.pt',
        data_path='models/graph_data.pt',
        mappings_path='models/company_mappings.pkl'
    )
    
    print("\n" + "="*70)
    print("SUPPLY CHAIN STRESS TESTING SYSTEM")
    print("="*70)
    
    # Run predefined scenarios
    scenarios = [
        CrisisScenarios.taiwan_earthquake(),
        CrisisScenarios.nvidia_supply_constraint(),
        CrisisScenarios.lithium_shortage(),
        CrisisScenarios.energy_crisis()
    ]
    
    results = {}
    for scenario in scenarios:
        df = tester.run_scenario(
            scenario['name'],
            scenario['affected'],
            scenario['intensity'],
            top_k=15
        )
        results[scenario['name']] = df
        
        # Visualize
        risk_scores = df.set_index('Company')['Risk_Score'].to_dict()
        fig = tester.visualize_impact(scenario['name'], risk_scores, top_k=20)
        
        # Save plot
        filename = scenario['name'].replace(' ', '_').replace('-', '_').lower()
        plt.savefig(f'{filename}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved visualization: {filename}.png")
    
    # Compare scenarios
    print("\n" + "="*70)
    print("SCENARIO COMPARISON")
    print("="*70)
    comparison = tester.compare_scenarios(scenarios)
    print("\nCompanies with highest average risk across all scenarios:")
    print(comparison.head(15).to_string(index=False))
    
    # Generate comprehensive report
    tester.generate_report(scenarios, 'stress_test_report.txt')
    
    print("\n✓ Stress testing complete!")


if __name__ == "__main__":
    main()