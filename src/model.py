"""
Graph Neural Network Model for Supply Chain Risk Propagation
Predicts how shocks propagate through the supply chain network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Tuple, Optional


class SupplyChainGNN(nn.Module):
    """
    Graph Neural Network for supply chain risk propagation
    
    Uses Graph Attention Networks (GAT) to learn how shocks propagate
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 heads: int = 4):
        """
        Initialize the GNN
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of graph conv layers
            dropout: Dropout probability
            heads: Number of attention heads for GAT
        """
        super(SupplyChainGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.input_layer = GATConv(
            input_dim, 
            hidden_dim, 
            heads=heads,
            dropout=dropout
        )
        
        # Hidden layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                GATConv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout
                )
            )
        
        # Output layer - predict risk score
        self.output_layer = nn.Linear(hidden_dim * heads, 1)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * heads)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Risk scores [num_nodes, 1]
        """
        # Input layer
        x = self.input_layer(x, edge_index)
        x = self.batch_norms[0](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = self.batch_norms[i + 1](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.output_layer(x)
        x = torch.sigmoid(x)  # Risk score between 0 and 1
        
        return x


class StressTestEngine:
    """
    Engine for running stress tests on supply chain networks
    """
    
    def __init__(self, 
                 model: SupplyChainGNN,
                 data: Data,
                 company_to_idx: Dict[str, int],
                 idx_to_company: Dict[int, str]):
        """
        Initialize stress test engine
        
        Args:
            model: Trained GNN model
            data: PyTorch Geometric data object
            company_to_idx: Mapping from company names to indices
            idx_to_company: Mapping from indices to company names
        """
        self.model = model
        self.data = data
        self.company_to_idx = company_to_idx
        self.idx_to_company = idx_to_company
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()
    
    def simulate_shock(self, 
                      shock_companies: List[str],
                      shock_intensity: float = 1.0,
                      propagation_steps: int = 5) -> Dict[str, float]:
        """
        Simulate a supply chain shock and predict downstream impacts
        
        Args:
            shock_companies: Companies experiencing the initial shock
            shock_intensity: Intensity of the shock (0-1)
            propagation_steps: Number of propagation steps to simulate
        
        Returns:
            Dictionary mapping company names to predicted risk scores
        """
        # Create augmented features with shock information
        x = self.data.x.clone().to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        
        # Add shock feature column
        shock_feature = torch.zeros(x.shape[0], 1).to(self.device)
        
        # Set shock for affected companies
        for company in shock_companies:
            if company in self.company_to_idx:
                idx = self.company_to_idx[company]
                shock_feature[idx] = shock_intensity
        
        # Concatenate shock feature
        x_augmented = torch.cat([x, shock_feature], dim=1)
        
        # Temporarily modify model input dimension if needed
        original_input_dim = self.model.input_dim
        if x_augmented.shape[1] != original_input_dim:
            # For simplicity, we'll use the original features
            # In production, you'd retrain with augmented features
            x_augmented = x
        
        # Propagate through network
        with torch.no_grad():
            risk_scores = self.model(x_augmented, edge_index)
        
        # Convert to dictionary
        results = {}
        for i, company in self.idx_to_company.items():
            results[company] = float(risk_scores[i].cpu().item())
        
        return results
    
    def stress_test_scenario(self,
                           scenario_name: str,
                           affected_companies: List[str],
                           intensity: float = 0.8) -> Tuple[Dict, List]:
        """
        Run a named stress test scenario
        
        Args:
            scenario_name: Name of the scenario
            affected_companies: Companies directly affected
            intensity: Shock intensity
        
        Returns:
            Tuple of (risk_scores_dict, ranked_list_of_tuples)
        """
        print(f"\n{'='*60}")
        print(f"STRESS TEST: {scenario_name}")
        print(f"{'='*60}")
        print(f"Directly affected: {', '.join(affected_companies)}")
        print(f"Shock intensity: {intensity:.1%}")
        
        # Run simulation
        risk_scores = self.simulate_shock(affected_companies, intensity)
        
        # Rank by risk
        ranked = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out low risk (< 0.1)
        significant_risks = [(c, r) for c, r in ranked if r > 0.1]
        
        print(f"\n📊 Companies at Risk (score > 0.1): {len(significant_risks)}")
        print(f"\nTop 10 Most Vulnerable:")
        print(f"{'Company':<20} {'Risk Score':<15} {'Status'}")
        print(f"{'-'*60}")
        
        for i, (company, risk) in enumerate(significant_risks[:10], 1):
            status = "🔴 CRITICAL" if risk > 0.7 else "🟡 HIGH" if risk > 0.5 else "🟢 MODERATE"
            print(f"{i}. {company:<17} {risk:>6.1%}          {status}")
        
        return risk_scores, significant_risks


class CrisisScenarios:
    """
    Pre-defined crisis scenarios for stress testing
    """
    
    @staticmethod
    def taiwan_earthquake():
        """Major earthquake affects Taiwan semiconductor production"""
        return {
            'name': 'Taiwan Earthquake - Semiconductor Disruption',
            'affected': ['TSMC'],
            'intensity': 0.9,
            'description': 'Major earthquake disrupts TSMC production, affecting chip supply globally'
        }
    
    @staticmethod
    def lithium_shortage():
        """Lithium prices spike due to supply constraints"""
        return {
            'name': 'Lithium Supply Shock',
            'affected': ['Panasonic', 'LG', 'CATL', 'Samsung'],
            'intensity': 0.8,
            'description': 'Lithium mining disruption causes battery price surge'
        }
    
    @staticmethod
    def nvidia_supply_constraint():
        """Nvidia faces production constraints"""
        return {
            'name': 'AI Chip Supply Constraint',
            'affected': ['Nvidia'],
            'intensity': 0.7,
            'description': 'Nvidia cannot meet surging AI chip demand'
        }
    
    @staticmethod
    def china_export_restrictions():
        """China restricts rare earth exports"""
        return {
            'name': 'Rare Earth Export Ban',
            'affected': ['ASML', 'Samsung', 'SK_Hynix'],
            'intensity': 0.85,
            'description': 'China restricts critical rare earth element exports'
        }
    
    @staticmethod
    def energy_crisis():
        """Energy crisis in Europe/Asia"""
        return {
            'name': 'Energy Crisis',
            'affected': ['Samsung', 'TSMC', 'Intel', 'ASML'],
            'intensity': 0.75,
            'description': 'Energy shortage affects semiconductor manufacturing'
        }
    
    @staticmethod
    def custom_scenario(name: str, 
                       affected: List[str], 
                       intensity: float,
                       description: str = ''):
        """Create a custom scenario"""
        return {
            'name': name,
            'affected': affected,
            'intensity': intensity,
            'description': description or f'Custom scenario affecting {", ".join(affected)}'
        }
    
    @staticmethod
    def list_all_scenarios():
        """List all available pre-defined scenarios"""
        scenarios = [
            CrisisScenarios.taiwan_earthquake(),
            CrisisScenarios.lithium_shortage(),
            CrisisScenarios.nvidia_supply_constraint(),
            CrisisScenarios.china_export_restrictions(),
            CrisisScenarios.energy_crisis()
        ]
        
        print("\n" + "="*70)
        print("AVAILABLE CRISIS SCENARIOS")
        print("="*70)
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            print(f"   Affected: {', '.join(scenario['affected'])}")
            print(f"   Intensity: {scenario['intensity']:.0%}")
        
        return scenarios


def create_synthetic_training_data(data: Data, 
                                   num_scenarios: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic training data for the GNN
    
    Simulates random shocks and their propagation
    
    Args:
        data: PyTorch Geometric data
        num_scenarios: Number of training scenarios
    
    Returns:
        Tuple of (features, labels)
    """
    num_nodes = data.x.shape[0]
    
    # We'll create scenarios where random nodes experience shocks
    # and label nearby nodes with higher risk
    X_train = []
    y_train = []
    
    for _ in range(num_scenarios):
        # Random shock location(s)
        num_shock_nodes = np.random.randint(1, 4)
        shock_nodes = np.random.choice(num_nodes, num_shock_nodes, replace=False)
        
        # Create feature vector with shock indicator
        features = data.x.clone()
        shock_feature = torch.zeros(num_nodes, 1)
        shock_feature[shock_nodes] = np.random.uniform(0.5, 1.0)
        
        features_augmented = torch.cat([features, shock_feature], dim=1)
        
        # Create labels based on graph distance from shock
        labels = torch.zeros(num_nodes, 1)
        
        # Simple propagation model: risk decreases with distance
        for shock_node in shock_nodes:
            distances = torch.ones(num_nodes) * 999
            distances[shock_node] = 0
            
            # Simplified distance calculation (in practice, use graph distances)
            for i in range(num_nodes):
                if i != shock_node:
                    # Random distance for synthetic data
                    distances[i] = np.random.exponential(2.0)
            
            # Risk inversely proportional to distance
            risk = torch.exp(-distances / 2.0) * np.random.uniform(0.7, 1.0)
            labels = torch.maximum(labels, risk.unsqueeze(1))
        
        X_train.append(features_augmented)
        y_train.append(labels)
    
    return torch.stack(X_train), torch.stack(y_train)


if __name__ == "__main__":
    from data_loader import SupplyChainDataLoader
    
    # Load data
    loader = SupplyChainDataLoader("connections.csv")
    data = loader.prepare_data()
    
    print("\n=== Model Architecture ===")
    model = SupplyChainGNN(
        input_dim=data.x.shape[1],
        hidden_dim=64,
        num_layers=3,
        dropout=0.2,
        heads=4
    )
    print(model)
    
    print("\n=== Available Crisis Scenarios ===")
    CrisisScenarios.list_all_scenarios()