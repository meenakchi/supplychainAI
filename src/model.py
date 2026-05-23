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
    Graph Attention Network for supply chain risk propagation.
    Uses learned attention weights to capture heterogeneous edge importance.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 heads: int = 4):
        """
        Args:
            input_dim:  Dimension of input node features (structural features + shock indicator)
            hidden_dim: Hidden layer dimension per head
            num_layers: Number of GAT conv layers (paper uses 3)
            dropout:    Dropout probability
            heads:      Number of attention heads
        """
        super(SupplyChainGNN, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout
        self.heads      = heads

        # Layer 1: input -> hidden*heads
        self.input_layer = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        # Layers 2..num_layers: hidden*heads -> hidden*heads
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                GATConv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
            )

        # Output projection: hidden*heads -> 1
        self.output_layer = nn.Linear(hidden_dim * heads, 1)

        # One BatchNorm per conv layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * heads)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Forward pass.

        Args:
            x:                Node features [num_nodes, input_dim]
            edge_index:       Edge indices  [2, num_edges]
            return_attention: If True, also return per-layer attention weights.

        Returns:
            risk_scores [num_nodes, 1] in [0, 1]
            (only when return_attention=True) list of (edge_idx, alpha) per layer,
            where alpha has shape [num_edges, heads] and edge_idx has shape [2, num_edges].
            Index 0 of the list is the input layer (used in the paper's analysis).
        """
        all_attentions = []

        # ── Layer 1 ──────────────────────────────────────────────────────────
        x, (ei0, a0) = self.input_layer(x, edge_index, return_attention_weights=True)
        if return_attention:
            all_attentions.append((ei0, a0))
        x = self.batch_norms[0](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Layers 2..num_layers ─────────────────────────────────────────────
        for i, conv in enumerate(self.conv_layers):
            x, (ei_i, a_i) = conv(x, edge_index, return_attention_weights=True)
            if return_attention:
                all_attentions.append((ei_i, a_i))
            x = self.batch_norms[i + 1](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Output ───────────────────────────────────────────────────────────
        x = torch.sigmoid(self.output_layer(x))

        if return_attention:
            # all_attentions[0] = input-layer weights (used in paper Section 4.4)
            return x, all_attentions
        return x


# ── Stress-test engine ────────────────────────────────────────────────────────

class StressTestEngine:
    """Runs scenario-level stress tests using the trained GNN."""

    def __init__(self,
                 model: SupplyChainGNN,
                 data: Data,
                 company_to_idx: Dict[str, int],
                 idx_to_company: Dict[int, str]):
        self.model           = model
        self.data            = data
        self.company_to_idx  = company_to_idx
        self.idx_to_company  = idx_to_company
        self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()

    def predict_impact(self,
                       shock_companies: List[str],
                       shock_intensity: float = 0.8) -> Dict[str, float]:
        x          = self.data.x.clone().to(self.device)
        edge_index = self.data.edge_index.to(self.device)

        shock = torch.zeros(x.shape[0], 1).to(self.device)
        for c in shock_companies:
            if c in self.company_to_idx:
                shock[self.company_to_idx[c]] = shock_intensity

        x_aug = torch.cat([x, shock], dim=1)
        with torch.no_grad():
            scores = self.model(x_aug, edge_index)

        return {self.idx_to_company[i]: float(scores[i].cpu().item())
                for i in self.idx_to_company}

    def run_scenario(self,
                     scenario_name: str,
                     affected_companies: List[str],
                     intensity: float = 0.8,
                     top_k: int = 15) -> "pd.DataFrame":
        import pandas as pd
        risk_scores = self.predict_impact(affected_companies, intensity)

        rows = []
        for company, risk in risk_scores.items():
            rows.append({
                'Company':      company,
                'Risk_Score':   risk,
                'Risk_Level':   self._categorise(risk),
                'Is_Epicenter': company in affected_companies,
            })

        df = pd.DataFrame(rows).sort_values('Risk_Score', ascending=False)
        print(f"\n{'='*60}\nSTRESS TEST: {scenario_name}\n{'='*60}")
        print(f"Epicenters: {', '.join(affected_companies)}  |  Intensity: {intensity:.0%}")
        print(f"\nTop {top_k}:")
        for _, row in df.head(top_k).iterrows():
            tag = '  <- EPICENTER' if row['Is_Epicenter'] else ''
            print(f"  {row['Company']:<22} {row['Risk_Score']:.3f}{tag}")
        return df

    @staticmethod
    def _categorise(score: float) -> str:
        if score > 0.70: return 'Critical'
        if score > 0.50: return 'High'
        if score > 0.30: return 'Moderate'
        return 'Low'


# ── Crisis scenario presets ───────────────────────────────────────────────────

class CrisisScenarios:
    """Pre-defined crisis scenarios matching paper Section 4.5."""

    @staticmethod
    def taiwan_earthquake():
        return {'name': 'Taiwan Earthquake', 'affected': ['TSMC'],
                'intensity': 0.90,
                'description': 'Major earthquake disrupts TSMC production'}

    @staticmethod
    def lithium_shortage():
        return {'name': 'Lithium Supply Shock',
                'affected': ['Panasonic', 'LG', 'CATL', 'Samsung'],
                'intensity': 0.80,
                'description': 'Battery material shortage drives cost inflation'}

    @staticmethod
    def nvidia_supply_constraint():
        return {'name': 'AI Chip Shortage', 'affected': ['Nvidia'],
                'intensity': 0.70,
                'description': 'Demand for AI accelerators exceeds Nvidia supply'}

    @staticmethod
    def china_export_restrictions():
        return {'name': 'Rare Earth Export Controls',
                'affected': ['ASML', 'Samsung', 'SK_Hynix'],
                'intensity': 0.85,
                'description': 'Export restrictions impact semiconductor tooling'}

    @staticmethod
    def energy_crisis():
        return {'name': 'Energy Supply Crisis',
                'affected': ['Samsung', 'TSMC', 'Intel', 'ASML'],
                'intensity': 0.75,
                'description': 'Power instability affects chip fabrication globally'}

    @staticmethod
    def port_disruption():
        return {'name': 'Port Disruption',
                'affected': ['Foxconn', 'Samsung', 'TSMC', 'Sony'],
                'intensity': 0.65,
                'description': 'Logistics bottlenecks delay manufacturing worldwide'}

    @staticmethod
    def custom_scenario(name: str, affected: List[str], intensity: float,
                        description: str = ''):
        return {'name': name, 'affected': affected, 'intensity': intensity,
                'description': description or f'Custom scenario: {", ".join(affected)}'}

    @staticmethod
    def list_all_scenarios():
        scenarios = [
            CrisisScenarios.taiwan_earthquake(),
            CrisisScenarios.lithium_shortage(),
            CrisisScenarios.nvidia_supply_constraint(),
            CrisisScenarios.china_export_restrictions(),
            CrisisScenarios.energy_crisis(),
            CrisisScenarios.port_disruption(),
        ]
        print('\n' + '='*70)
        print('AVAILABLE CRISIS SCENARIOS')
        print('='*70)
        for i, s in enumerate(scenarios, 1):
            print(f"\n{i}. {s['name']}")
            print(f"   {s['description']}")
            print(f"   Affected:  {', '.join(s['affected'])}")
            print(f"   Intensity: {s['intensity']:.0%}")
        return scenarios


if __name__ == '__main__':
    from data_loader import SupplyChainDataLoader
    loader = SupplyChainDataLoader('data/connections.csv')
    data   = loader.prepare_data()

    model = SupplyChainGNN(
        input_dim=data.x.shape[1] + 1,
        hidden_dim=64, num_layers=3, dropout=0.2, heads=4,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')
    CrisisScenarios.list_all_scenarios()