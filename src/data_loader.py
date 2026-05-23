"""
Data Loader for Supply Chain Network
Loads connections.csv and builds the directed graph + PyG Data object.

Edge direction convention (consistent with paper Section 3.1):
    Edge (u -> v) means u SUPPLIES v, so risk propagates u -> v.

CSV semantics:
    source, target, Supplier  =>  target supplies source  =>  add_edge(target, source)
    source, target, Customer  =>  target supplies source  =>  add_edge(target, source)
Both cases: supplier is always `target`, customer is always `source`.
"""

import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
from typing import Dict, List, Tuple, Optional


class SupplyChainDataLoader:
    """Load and prepare supply chain network data for GNN training."""

    def __init__(self, connections_path: str):
        self.connections_path = connections_path
        self.df               = None
        self.graph            = None          # nx.DiGraph, edges = supplier -> customer
        self.company_to_idx: Dict[str, int] = {}
        self.idx_to_company: Dict[int, str] = {}
        self.node_features: Optional[torch.Tensor] = None

    # ── Public pipeline ───────────────────────────────────────────────────────

    def prepare_data(self) -> Data:
        """Full pipeline: load CSV → build graph → features → PyG Data."""
        self.load_connections()
        self.build_networkx_graph()
        self.create_node_mapping()
        self.compute_node_features()
        data = self.to_pytorch_geometric()
        print(f'\nData ready: {len(self.company_to_idx)} companies, '
              f'{self.graph.number_of_edges()} edges, '
              f'{self.node_features.shape[1]} features per node')
        return data

    # ── Steps ─────────────────────────────────────────────────────────────────

    def load_connections(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.connections_path)
        print(f'Loaded {len(self.df)} rows from {self.connections_path}')
        return self.df

    def build_networkx_graph(self) -> nx.DiGraph:
        """
        Build directed graph where every edge points supplier -> customer
        so that risk propagation follows edge direction naturally.

        CSV rows:
          "Nvidia, TSMC, Supplier"  => TSMC supplies Nvidia => edge TSMC -> Nvidia
          "Microsoft, Nvidia, Customer" => Nvidia supplies Microsoft => edge Nvidia -> Microsoft
        Both cases: add_edge(target, source)
        """
        self.graph = nx.DiGraph()
        for _, row in self.df.iterrows():
            src, tgt = row['source'], row['target']
            # In both Supplier and Customer rows, `target` is the upstream supplier
            self.graph.add_edge(tgt, src)

        print(f'Graph: {self.graph.number_of_nodes()} nodes, '
              f'{self.graph.number_of_edges()} edges')
        return self.graph

    def create_node_mapping(self) -> Tuple[Dict, Dict]:
        companies = list(self.graph.nodes())
        self.company_to_idx = {c: i for i, c in enumerate(companies)}
        self.idx_to_company = {i: c for c, i in self.company_to_idx.items()}
        return self.company_to_idx, self.idx_to_company

    def compute_node_features(self) -> torch.Tensor:
        """
        Structural node features (paper Table 1):
            0: in-degree          (number of suppliers)
            1: out-degree         (number of customers)
            2: PageRank           (global importance)
            3: betweenness centrality (bridge position)
            4: clustering coefficient (local redundancy)
        All standardised to μ=0, σ=1.
        The shock indicator is NOT included here; it is concatenated at training time.
        """
        n = self.graph.number_of_nodes()
        features = np.zeros((n, 5), dtype=np.float32)

        in_deg      = dict(self.graph.in_degree())
        out_deg     = dict(self.graph.out_degree())
        pagerank    = nx.pagerank(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)
        clustering  = nx.clustering(self.graph.to_undirected())

        for company, idx in self.company_to_idx.items():
            features[idx, 0] = in_deg.get(company, 0)
            features[idx, 1] = out_deg.get(company, 0)
            features[idx, 2] = pagerank.get(company, 0.0)
            features[idx, 3] = betweenness.get(company, 0.0)
            features[idx, 4] = clustering.get(company, 0.0)

        # Standardise: μ=0, σ=1 per feature; +1e-8 prevents div-by-zero on zero-variance cols
        means = features.mean(axis=0)
        stds  = features.std(axis=0)
        features = (features - means) / (stds + 1e-8)

        self.node_features = torch.FloatTensor(features)
        return self.node_features

    def to_pytorch_geometric(self) -> Data:
        """Convert to PyG Data, aligning node indices with company_to_idx."""
        G_relabeled = nx.relabel_nodes(self.graph, self.company_to_idx)
        data = from_networkx(G_relabeled)
        data.x = self.node_features
        data.company_names = [self.idx_to_company[i]
                              for i in range(len(self.idx_to_company))]
        return data

    # ── Utility helpers ───────────────────────────────────────────────────────

    def get_company_neighbors(self, company: str,
                              direction: str = 'both') -> List[str]:
        """
        Return suppliers ('in'), customers ('out'), or both.
        In our directed graph: predecessors = suppliers, successors = customers.
        """
        if company not in self.graph:
            return []
        neighbors = []
        if direction in ('suppliers', 'in', 'both'):
            neighbors.extend(self.graph.predecessors(company))
        if direction in ('customers', 'out', 'both'):
            neighbors.extend(self.graph.successors(company))
        return list(set(neighbors))

    def get_dependency_path(self, source: str,
                            target: str) -> Optional[List[str]]:
        """Shortest directed path from source to target (following risk flow)."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None


if __name__ == '__main__':
    loader = SupplyChainDataLoader('data/connections.csv')
    data   = loader.prepare_data()

    for company in ['Apple', 'TSMC', 'Nvidia']:
        suppliers = loader.get_company_neighbors(company, 'suppliers')
        customers = loader.get_company_neighbors(company, 'customers')
        print(f'\n{company}:')
        print(f'  Suppliers ({len(suppliers)}): {suppliers[:5]}')
        print(f'  Customers ({len(customers)}): {customers[:5]}')