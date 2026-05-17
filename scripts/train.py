"""
Training Script for Supply Chain GNN
Trains the model on synthetic shock propagation data
"""

import sys
import os
import pickle
from typing import Tuple, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from torch_geometric.data import Data

from data_loader import SupplyChainDataLoader
from model import SupplyChainGNN


class ShockPropagationSimulator:
    """
    Simulates how shocks propagate through supply chains.
    Creates synthetic training targets based on graph structure.
    """

    def __init__(self, graph: nx.DiGraph, company_to_idx: dict, device: str = "cpu"):
        self.graph = graph
        self.company_to_idx = company_to_idx
        self.idx_to_company = {v: k for k, v in company_to_idx.items()}
        self.num_nodes = len(company_to_idx)
        self.device = device

    def simulate_shock_propagation(self, shock_nodes: List[int], shock_intensity: float) -> torch.Tensor:
        scores = {n: 0.0 for n in self.graph.nodes}
        company_nodes = [self.idx_to_company[i] for i in shock_nodes if i in self.idx_to_company]

        for n in company_nodes:
            if n in scores:
                scores[n] = float(shock_intensity)

        DECAY_DOWN, DECAY_UP = 0.72, 0.42

        queue, visited = list(company_nodes), set(company_nodes)
        while queue:
            node = queue.pop(0)
            if scores[node] < 0.04:
                continue
            for c in self.graph.successors(node):
                prop = min(scores[node] * DECAY_DOWN * np.random.uniform(0.75, 1.15), 1.0)
                if prop > scores.get(c, 0.0):
                    scores[c] = prop
                    if c not in visited:
                        visited.add(c)
                        queue.append(c)

        queue, visited = list(company_nodes), set(company_nodes)
        while queue:
            node = queue.pop(0)
            if scores[node] < 0.04:
                continue
            for s in self.graph.predecessors(node):
                prop = min(scores[node] * DECAY_UP * np.random.uniform(0.6, 1.0), 1.0)
                if prop > scores.get(s, 0.0):
                    scores[s] = prop
                    if s not in visited:
                        visited.add(s)
                        queue.append(s)

        result = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
        for company, score in scores.items():
            if company in self.company_to_idx:
                result[self.company_to_idx[company]] = float(score)
        return result

    def generate_training_batch(
        self,
        batch_size: int,
        base_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = base_features.to(self.device).float()
        num_nodes, _ = x.shape

        X_batch = []
        y_batch = []

        for _ in range(batch_size):
            num_shocks = np.random.randint(1, min(4, num_nodes) + 1)
            shock_nodes = np.random.choice(num_nodes, num_shocks, replace=False)
            shock_intensity = float(np.random.uniform(0.6, 1.0))

            shock_feature = torch.zeros(num_nodes, 1, dtype=torch.float32, device=self.device)
            shock_feature[torch.tensor(shock_nodes, dtype=torch.long, device=self.device)] = shock_intensity

            features = torch.cat([x, shock_feature], dim=1)
            risk_scores = self.simulate_shock_propagation(shock_nodes.tolist(), shock_intensity)

            X_batch.append(features)
            y_batch.append(risk_scores.unsqueeze(1))

        return torch.stack(X_batch), torch.stack(y_batch)


class GNNTrainer:
    """
    Trainer for the Supply Chain GNN.
    """

    def __init__(
        self,
        model: SupplyChainGNN,
        data: Data,
        simulator: ShockPropagationSimulator,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.data = data
        self.simulator = simulator
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, num_batches: int, batch_size: int) -> float:
        self.model.train()
        total_loss = 0.0
        edge_index = self.data.edge_index.to(self.device)

        for _ in range(num_batches):
            X_batch, y_batch = self.simulator.generate_training_batch(batch_size, self.data.x)

            batch_loss = 0.0
            for i in range(batch_size):
                x = X_batch[i].to(self.device)
                y_true = y_batch[i].to(self.device)

                y_pred = self.model(x, edge_index)
                loss = self.criterion(y_pred, y_true)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item()

            total_loss += batch_loss / batch_size

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, num_batches: int, batch_size: int) -> float:
        self.model.eval()
        total_loss = 0.0
        edge_index = self.data.edge_index.to(self.device)

        for _ in range(num_batches):
            X_batch, y_batch = self.simulator.generate_training_batch(batch_size, self.data.x)

            batch_loss = 0.0
            for i in range(batch_size):
                x = X_batch[i].to(self.device)
                y_true = y_batch[i].to(self.device)

                y_pred = self.model(x, edge_index)
                loss = self.criterion(y_pred, y_true)
                batch_loss += loss.item()

            total_loss += batch_loss / batch_size

        return total_loss / num_batches

    def train(
        self,
        num_epochs: int = 50,
        batches_per_epoch: int = 20,
        batch_size: int = 16,
        val_batches: int = 10,
        early_stopping_patience: int = 10
    ) -> dict:
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float("inf")
        patience_counter = 0

        os.makedirs("models", exist_ok=True)

        for epoch in tqdm(range(num_epochs), desc="Training"):
            train_loss = self.train_epoch(batches_per_epoch, batch_size)
            val_loss = self.validate(val_batches, batch_size)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("models/best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        self.load_checkpoint("models/best_model.pt")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
        }

    def save_checkpoint(self, filepath: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", self.train_losses)
        self.val_losses = checkpoint.get("val_losses", self.val_losses)

    def plot_training_history(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.train_losses, label="Train Loss", linewidth=2)
        ax.plot(self.val_losses, label="Validation Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.set_title("Training History", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def main():
    print("=" * 60)
    print("SUPPLY CHAIN GNN TRAINING")
    print("=" * 60)

    print("\n1. Loading supply chain data...")
    loader = SupplyChainDataLoader(os.path.join(os.path.dirname(__file__), "..", "data", "connections.csv"))
    data = loader.prepare_data()

    data.x = data.x.float()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n2. Creating shock propagation simulator...")
    simulator = ShockPropagationSimulator(loader.graph, loader.company_to_idx, device=device)

    print("\n3. Initializing GNN model...")
    model = SupplyChainGNN(
        input_dim=data.x.shape[1] + 1,
        hidden_dim=64,
        num_layers=3,
        dropout=0.2,
        heads=4,
    )

    trainer = GNNTrainer(model, data, simulator, device)

    print("\n4. Training model...")
    history = trainer.train(
        num_epochs=50,
        batches_per_epoch=20,
        batch_size=16,
        val_batches=10,
        early_stopping_patience=10,
    )

    print("\n5. Plotting training history...")
    fig = trainer.plot_training_history()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved training_history.png")

    print("\n6. Saving model and metadata...")
    os.makedirs("models", exist_ok=True)

    trainer.save_checkpoint("models/supply_chain_gnn.pt")

    with open("models/company_mappings.pkl", "wb") as f:
        pickle.dump(
            {
                "company_to_idx": loader.company_to_idx,
                "idx_to_company": loader.idx_to_company,
            },
            f,
        )

    torch.save(data, "models/graph_data.pt")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print("Model saved to: models/supply_chain_gnn.pt")
    print("Data saved to: models/graph_data.pt")
    print("Mappings saved to: models/company_mappings.pkl")


if __name__ == "__main__":
    main()
    