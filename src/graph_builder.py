"""
Graph Builder and Visualizer
Advanced graph analysis and visualization for supply chain networks
"""

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd


class SupplyChainGraphAnalyzer:
    """Advanced graph analysis and visualization"""
    
    def __init__(self, graph: nx.DiGraph, company_to_idx: Dict):
        """
        Initialize graph analyzer
        
        Args:
            graph: NetworkX directed graph
            company_to_idx: Mapping from company names to indices
        """
        self.graph = graph
        self.company_to_idx = company_to_idx
        self.idx_to_company = {v: k for k, v in company_to_idx.items()}
        
    def identify_critical_nodes(self, top_k: int = 10) -> pd.DataFrame:
        """
        Identify most critical nodes in the supply chain
        
        Args:
            top_k: Number of top critical nodes to return
        
        Returns:
            DataFrame with critical nodes and their metrics
        """
        # Calculate various centrality measures
        pagerank = nx.pagerank(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        
        # Create dataframe
        companies = list(self.graph.nodes())
        data = {
            'Company': companies,
            'PageRank': [pagerank[c] for c in companies],
            'Betweenness': [betweenness[c] for c in companies],
            'Suppliers_Count': [in_degree[c] for c in companies],
            'Customers_Count': [out_degree[c] for c in companies],
            'Total_Connections': [in_degree[c] + out_degree[c] for c in companies]
        }
        
        df = pd.DataFrame(data)
        df = df.sort_values('PageRank', ascending=False)
        
        return df.head(top_k)
    
    def find_vulnerability_clusters(self) -> List[List[str]]:
        """
        Identify clusters of companies that are highly interdependent
        
        Returns:
            List of clusters (each cluster is a list of companies)
        """
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Find strongly connected components in directed graph
        # These are groups that can reach each other
        components = list(nx.strongly_connected_components(self.graph))
        
        # Filter for non-trivial components (size > 1)
        clusters = [list(comp) for comp in components if len(comp) > 1]
        
        return clusters
    
    def calculate_supply_chain_depth(self, company: str) -> Dict[str, int]:
        """
        Calculate how many hops each company is from the given company
        
        Args:
            company: Root company
        
        Returns:
            Dictionary mapping companies to their distance
        """
        if company not in self.graph:
            return {}
        
        # Downstream (customers) - how far can impact travel
        try:
            downstream_distances = nx.single_source_shortest_path_length(
                self.graph, company
            )
        except:
            downstream_distances = {company: 0}
        
        # Upstream (suppliers) - how far back do dependencies go
        try:
            reversed_graph = self.graph.reverse()
            upstream_distances = nx.single_source_shortest_path_length(
                reversed_graph, company
            )
        except:
            upstream_distances = {company: 0}
        
        return {
            'downstream': downstream_distances,
            'upstream': upstream_distances
        }
    
    def visualize_network(self, 
                         highlight_companies: Optional[List[str]] = None,
                         layout: str = 'spring',
                         figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Visualize the supply chain network
        
        Args:
            highlight_companies: Companies to highlight
            layout: Layout algorithm ('spring', 'kamada_kawai', 'circular')
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.circular_layout(self.graph)
        
        # Calculate node sizes based on importance
        pagerank = nx.pagerank(self.graph)
        node_sizes = [pagerank[node] * 10000 for node in self.graph.nodes()]
        
        # Color nodes
        node_colors = []
        for node in self.graph.nodes():
            if highlight_companies and node in highlight_companies:
                node_colors.append('#FF6B6B')  # Red for highlighted
            else:
                node_colors.append('#4ECDC4')  # Teal for normal
        
        # Draw network
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            alpha=0.3,
            arrows=True,
            arrowsize=10,
            width=1.5,
            ax=ax
        )
        
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title("Supply Chain Network", fontsize=20, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def visualize_interactive(self, 
                            highlight_companies: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive Plotly visualization
        
        Args:
            highlight_companies: Companies to highlight
        
        Returns:
            Plotly figure
        """
        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Calculate node metrics
        pagerank = nx.pagerank(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create hover text
            suppliers = list(self.graph.predecessors(node))
            customers = list(self.graph.successors(node))
            text = f"<b>{node}</b><br>"
            text += f"PageRank: {pagerank[node]:.4f}<br>"
            text += f"Betweenness: {betweenness[node]:.4f}<br>"
            text += f"Suppliers: {len(suppliers)}<br>"
            text += f"Customers: {len(customers)}"
            node_text.append(text)
            
            # Color
            if highlight_companies and node in highlight_companies:
                node_color.append('#FF6B6B')
            else:
                node_color.append('#4ECDC4')
            
            # Size based on PageRank
            node_size.append(pagerank[node] * 500 + 10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node for node in self.graph.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Interactive Supply Chain Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800
            )
        )
        
        return fig
    
    def analyze_bottlenecks(self) -> pd.DataFrame:
        """
        Identify potential bottlenecks in the supply chain
        
        Returns:
            DataFrame with bottleneck analysis
        """
        betweenness = nx.betweenness_centrality(self.graph)
        
        # Find articulation points (nodes whose removal disconnects the graph)
        undirected = self.graph.to_undirected()
        articulation_points = set(nx.articulation_points(undirected))
        
        # Create analysis dataframe
        companies = list(self.graph.nodes())
        data = {
            'Company': companies,
            'Betweenness': [betweenness[c] for c in companies],
            'Is_Articulation_Point': [c in articulation_points for c in companies],
            'In_Degree': [self.graph.in_degree(c) for c in companies],
            'Out_Degree': [self.graph.out_degree(c) for c in companies]
        }
        
        df = pd.DataFrame(data)
        df = df.sort_values('Betweenness', ascending=False)
        
        return df


if __name__ == "__main__":
    # Test with sample data
    from data_loader import SupplyChainDataLoader
    
    loader = SupplyChainDataLoader("connections.csv")
    data = loader.prepare_data()
    
    analyzer = SupplyChainGraphAnalyzer(loader.graph, loader.company_to_idx)
    
    print("\n=== Critical Nodes ===")
    print(analyzer.identify_critical_nodes(10))
    
    print("\n=== Bottleneck Analysis ===")
    print(analyzer.analyze_bottlenecks().head(10))
    
    print("\n=== Vulnerability Clusters ===")
    clusters = analyzer.find_vulnerability_clusters()
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")