"""
Training Script for Supply Chain GNN
Trains the model on synthetic shock propagation data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

from data_loader import SupplyChainDataLoader
from model import SupplyChainGNN
import networkx as nx


class ShockPropagationSimulator:
    """
    Simulates how shocks propagate through supply chains
    Creates realistic training data based on graph structure
    """
    
    def __init__(self, graph: nx.DiGraph, company_to_idx: dict):
        self.graph = graph
        self.company_to_idx = company_to_idx
        self.idx_to_company = {v: k for k, v in company_to_idx.items()}
        self.num_nodes = len(company_to_idx)
        
    def simulate_shock_propagation(self, 
                                   shock_nodes: List[int],
                                   shock_intensity: float) -> torch.Tensor:
        """
        Simulate how a shock propagates through the network
        
        Args:
            shock_nodes: Indices of nodes experiencing shock
            shock_intensity: Initial shock intensity (0-1)
        
        Returns:
            Risk scores for all nodes
        """
        # Initialize risk scores
        risk_scores = np.zeros(self.num_nodes)
        
        # Set initial shock
        for node_idx in shock_nodes:
            risk_scores[node_idx] = shock_intensity
        
        # Propagate through network using shortest path distances
        for shock_idx in shock_nodes:
            shock_company = self.idx_to_company[shock_idx]
            
            # Calculate distances to all other nodes
            try:
                # Downstream propagation (to customers)
                downstream_lengths = nx.single_source_shortest_path_length(
                    self.graph, shock_company
                )
                
                for target_company, distance in downstream_lengths.items():
                    target_idx = self.company_to_idx[target_company]
                    
                    # Risk decays with distance
                    # Add some randomness for realism
                    decay_factor = np.exp(-distance / 2.0)
                    propagated_risk = shock_intensity * decay_factor * np.random.uniform(0.7, 1.0)
                    
                    # Take maximum risk if multiple paths
                    risk_scores[target_idx] = max(risk_scores[target_idx], propagated_risk)
            except:
                pass
            
            # Upstream propagation (to suppliers) - weaker effect
            try:
                reversed_graph = self.graph.reverse()
                upstream_lengths = nx.single_source_shortest_path_length(
                    reversed_graph, shock_company
                )
                
                for target_company, distance in upstream_lengths.items():
                    target_idx = self.company_to_idx[target_company]
                    
                    # Upstream risk is weaker (demand shock)
                    decay_factor = np.exp(-distance / 3.0)
                    propagated_risk = shock_intensity * decay_factor * 0.5 * np.random.uniform(0.5, 1.0)
                    
                    risk_scores[target_idx] = max(risk_scores[target_idx], propagated_risk)
            except:
                pass
        
        return torch.FloatTensor(risk_scores)
    
    def generate_training_batch(self, 
                               batch_size: int,
                               base_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of training examples
        
        Args:
            batch_size: Number of examples to generate
            base_features: Base node features
        
        Returns:
            Tuple of (augmented_features, risk_labels)
        """
        X_batch = []
        y_batch = []
        
        for _ in range(batch_size):
            # Random number of shock sources (1-3)
            num_shocks = np.random.randint(1, 4)
            shock_nodes = np.random.choice(self.num_nodes, num_shocks, replace=False)
            shock_intensity = np.random.uniform(0.6, 1.0)
            
            # Create shock indicator feature
            shock_feature = torch.zeros(self.num_nodes, 1)
            for node_idx in shock_nodes:
                shock_feature[node_idx] = shock_intensity
            
            # Augment base features with shock indicator
            features = torch.cat([base_features, shock_feature], dim=1)
            
            # Simulate propagation
            risk_scores = self.simulate_shock_propagation(shock_nodes, shock_intensity)
            
            X_batch.append(features)
            y_batch.append(risk_scores.unsqueeze(1))
        
        return torch.stack(X_batch), torch.stack(y_batch)


class GNNTrainer:
    """
    Trainer for the Supply Chain GNN
    """
    
    def __init__(self, 
                 model: SupplyChainGNN,
                 data: Data,
                 simulator: ShockPropagationSimulator,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.data = data
        self.simulator = simulator
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, num_batches: int, batch_size: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        edge_index = self.data.edge_index.to(self.device)
        
        for _ in range(num_batches):
            # Generate batch
            X_batch, y_batch = self.simulator.generate_training_batch(
                batch_size, self.data.x
            )
            
            batch_loss = 0.0
            for i in range(batch_size):
                # Forward pass
                x = X_batch[i].to(self.device)
                y_true = y_batch[i].to(self.device)
                
                y_pred = self.model(x, edge_index)
                
                # Compute loss
                loss = self.criterion(y_pred, y_true)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                batch_loss += loss.item()
            
            total_loss += batch_loss / batch_size
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, num_batches: int, batch_size: int) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        edge_index = self.data.edge_index.to(self.device)
        
        with torch.no_grad():
            for _ in range(num_batches):
                X_batch, y_batch = self.simulator.generate_training_batch(
                    batch_size, self.data.x
                )
                
                batch_loss = 0.0
                for i in range(batch_size):
                    x = X_batch[i].to(self.device)
                    y_true = y_batch[i].to(self.device)
                    
                    y_pred = self.model(x, edge_index)
                    loss = self.criterion(y_pred, y_true)
                    
                    batch_loss += loss.item()
                
                total_loss += batch_loss / batch_size
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, 
             num_epochs: int = 50,
             batches_per_epoch: int = 20,
             batch_size: int = 16,
             val_batches: int = 10,
             early_stopping_patience: int = 10) -> dict:
        """
        Complete training loop
        
        Args:
            num_epochs: Number of training epochs
            batches_per_epoch: Batches per epoch
            batch_size: Batch size
            val_batches: Validation batches
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history
        """
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Train
            train_loss = self.train_epoch(batches_per_epoch, batch_size)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_batches, batch_size)
            self.val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.load_checkpoint('best_model.pt')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.train_losses, label='Train Loss', linewidth=2)
        ax.plot(self.val_losses, label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def main():
    """Main training script"""
    print("="*60)
    print("SUPPLY CHAIN GNN TRAINING")
    print("="*60)
    
    # Load data
    print("\n1. Loading supply chain data...")
    loader = SupplyChainDataLoader("../data/connections.csv")
    data = loader.prepare_data()
    
    # Create simulator
    print("\n2. Creating shock propagation simulator...")
    simulator = ShockPropagationSimulator(loader.graph, loader.company_to_idx)
    
    # Initialize model
    print("\n3. Initializing GNN model...")
    # Add 1 to input_dim for shock indicator feature
    model = SupplyChainGNN(
        input_dim=data.x.shape[1] + 1,  # +1 for shock indicator
        hidden_dim=64,
        num_layers=3,
        dropout=0.2,
        heads=4
    )
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = GNNTrainer(model, data, simulator, device)
    
    # Train
    print("\n4. Training model...")
    history = trainer.train(
        num_epochs=50,
        batches_per_epoch=20,
        batch_size=16,
        val_batches=10,
        early_stopping_patience=10
    )
    
    # Plot results
    print("\n5. Plotting training history...")
    fig = trainer.plot_training_history()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Saved training_history.png")
    
    # Save final model and metadata
    print("\n6. Saving model and metadata...")
    os.makedirs('models', exist_ok=True)
    
    # Save model
    trainer.save_checkpoint('models/supply_chain_gnn.pt')
    
    # Save mappings
    with open('models/company_mappings.pkl', 'wb') as f:
        pickle.dump({
            'company_to_idx': loader.company_to_idx,
            'idx_to_company': loader.idx_to_company
        }, f)
    
    # Save data
    torch.save(data, 'models/graph_data.pt')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Model saved to: models/supply_chain_gnn.pt")
    print(f"Data saved to: models/graph_data.pt")
    print(f"Mappings saved to: models/company_mappings.pkl")


if __name__ == "__main__":
    main()