"""
Data Loader for Supply Chain Network
Loads connections and builds the graph structure
"""

import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
from typing import Dict, List, Tuple, Optional


class SupplyChainDataLoader:
    """Load and prepare supply chain network data"""
    
    def __init__(self, connections_path: str):
        """
        Initialize the data loader
        
        Args:
            connections_path: Path to CSV file with supply chain connections
        """
        self.connections_path = connections_path
        self.df = None
        self.graph = None
        self.company_to_idx = {}
        self.idx_to_company = {}
        self.node_features = None
        
    def load_connections(self) -> pd.DataFrame:
        """Load the connections CSV file"""
        self.df = pd.read_csv(self.connections_path)
        print(f"Loaded {len(self.df)} connections")
        return self.df
    
    def build_networkx_graph(self) -> nx.DiGraph:
        """Build a directed graph from connections"""
        self.graph = nx.DiGraph()
        
        # Add edges with relationship types
        for _, row in self.df.iterrows():
            source = row['source']
            target = row['target']
            relationship = row['relationship']
            
            # Add edge (direction matters!)
            # Supplier -> Customer means dependency flows downstream
            if relationship == 'Supplier':
                # If A is supplier to B, then B depends on A
                self.graph.add_edge(source, target, relationship=relationship)
            elif relationship == 'Customer':
                # If A is customer of B, then A depends on B
                self.graph.add_edge(target, source, relationship='Supplier')
        
        print(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def create_node_mapping(self) -> Tuple[Dict, Dict]:
        """Create bidirectional mapping between company names and indices"""
        companies = list(self.graph.nodes())
        self.company_to_idx = {company: idx for idx, company in enumerate(companies)}
        self.idx_to_company = {idx: company for company, idx in self.company_to_idx.items()}
        return self.company_to_idx, self.idx_to_company
    
    def compute_node_features(self) -> torch.Tensor:
        """
        Compute initial node features based on graph structure
        
        Features:
        - In-degree (how many suppliers)
        - Out-degree (how many customers)
        - PageRank (importance in network)
        - Betweenness centrality (bridge position)
        - Clustering coefficient (how connected neighbors are)
        """
        num_nodes = self.graph.number_of_nodes()
        features = np.zeros((num_nodes, 5))
        
        # Compute centrality metrics
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        pagerank = nx.pagerank(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)
        
        # For directed graphs, clustering coefficient needs undirected version
        undirected = self.graph.to_undirected()
        clustering = nx.clustering(undirected)
        
        # Fill feature matrix
        for company, idx in self.company_to_idx.items():
            features[idx, 0] = in_degree.get(company, 0)
            features[idx, 1] = out_degree.get(company, 0)
            features[idx, 2] = pagerank.get(company, 0)
            features[idx, 3] = betweenness.get(company, 0)
            features[idx, 4] = clustering.get(company, 0)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        self.node_features = torch.FloatTensor(features)
        return self.node_features
    
    def to_pytorch_geometric(self) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object
        
        Returns:
            PyTorch Geometric Data object
        """
        # Relabel nodes to use integer indices
        mapping = {company: idx for idx, company in enumerate(self.graph.nodes())}
        G_relabeled = nx.relabel_nodes(self.graph, mapping)
        
        # Convert to PyTorch Geometric
        data = from_networkx(G_relabeled)
        
        # Add node features
        data.x = self.node_features
        
        # Store company names as metadata
        data.company_names = [self.idx_to_company[i] for i in range(len(self.idx_to_company))]
        
        return data
    
    def get_company_neighbors(self, company: str, direction: str = 'both') -> List[str]:
        """
        Get suppliers or customers of a company
        
        Args:
            company: Company name
            direction: 'suppliers' (in-edges), 'customers' (out-edges), or 'both'
        
        Returns:
            List of connected companies
        """
        if company not in self.graph:
            return []
        
        neighbors = []
        
        if direction in ['suppliers', 'both']:
            # Predecessors are suppliers (incoming edges)
            neighbors.extend(list(self.graph.predecessors(company)))
        
        if direction in ['customers', 'both']:
            # Successors are customers (outgoing edges)
            neighbors.extend(list(self.graph.successors(company)))
        
        return list(set(neighbors))
    
    def get_dependency_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find shortest dependency path between two companies
        
        Args:
            source: Starting company
            target: Target company
        
        Returns:
            List of companies in the path, or None if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def prepare_data(self) -> Data:
        """
        Complete data preparation pipeline
        
        Returns:
            PyTorch Geometric Data object ready for GNN
        """
        self.load_connections()
        self.build_networkx_graph()
        self.create_node_mapping()
        self.compute_node_features()
        data = self.to_pytorch_geometric()
        
        print(f"\nData preparation complete!")
        print(f"Number of companies: {len(self.company_to_idx)}")
        print(f"Feature dimension: {self.node_features.shape[1]}")
        
        return data


if __name__ == "__main__":
    # Test the data loader
    loader = SupplyChainDataLoader("connections.csv")
    data = loader.prepare_data()
    
    print("\nSample company neighbors:")
    for company in ["Apple", "TSMC", "Nvidia"]:
        suppliers = loader.get_company_neighbors(company, "suppliers")
        customers = loader.get_company_neighbors(company, "customers")
        print(f"\n{company}:")
        print(f"  Suppliers: {suppliers}")
        print(f"  Customers: {customers}")