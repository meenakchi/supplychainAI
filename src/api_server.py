"""
FastAPI Backend — Supply Chain Risk Analyzer
Fixed version: loads real graph from connections.csv, runs proper BFS propagation.
GNN model loaded if available; falls back to graph-based simulation otherwise.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import os, sys, pickle, json
import numpy as np
import pandas as pd
import networkx as nx

# ── Optional GNN import ───────────────────────────────────────────────────────
try:
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from model import SupplyChainGNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

app = FastAPI(
    title="Supply Chain Risk Analyzer",
    description="Graph-based supply chain risk propagation with optional GNN inference.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ShockScenario(BaseModel):
    affected_companies: List[str] = Field(..., description="Epicentre companies")
    intensity: float             = Field(0.8, ge=0.0, le=1.0)
    scenario_name: Optional[str] = None
    monte_carlo_n: int           = Field(0, ge=0, le=2000,
                                         description="If >0, run MC and return percentiles")


class NodeRisk(BaseModel):
    company:    str
    risk_score: float
    risk_level: str
    is_epicenter: bool
    p5:  Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    std: Optional[float] = None


class ScenarioResult(BaseModel):
    scenario_name:      str
    affected_companies: List[str]
    intensity:          float
    inference_mode:     str          # "gnn" | "bfs"
    timestamp:          str
    predictions:        List[NodeRisk]
    summary:            Dict[str, int]


# ── App state ─────────────────────────────────────────────────────────────────

state: Dict = {
    "graph":           None,   # nx.DiGraph
    "company_to_idx":  None,
    "idx_to_company":  None,
    "gnn_model":       None,
    "gnn_data":        None,
    "device":          None,
}

CONNECTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "connections.csv"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Graph helpers ─────────────────────────────────────────────────────────────

def load_graph(path: str) -> nx.DiGraph:
    """Build directed supply-chain graph from CSV.
 
    CSV columns: source, target, relationship
    
    INTERPRETATION:
      "Nvidia, TSMC, Supplier"  => TSMC is Nvidia's supplier => TSMC supplies Nvidia
      Edge direction (risk flow): TSMC --> Nvidia
      So: add_edge(target, source)  i.e. add_edge(TSMC, Nvidia)
 
      "Microsoft, Nvidia, Customer" => Microsoft is Nvidia's customer => Nvidia supplies Microsoft
      Edge direction (risk flow): Nvidia --> Microsoft
      So: add_edge(target, source)  i.e. add_edge(Nvidia, Microsoft)
 
    Both cases reduce to the same rule: add_edge(target, source).
    The original api_server.py was wrong for Supplier rows (it did src->tgt).
    """
    df = pd.read_csv(path)
    G = nx.DiGraph()
 
    for _, row in df.iterrows():
        src, tgt, rel = row["source"], row["target"], row["relationship"]
 
        if rel == "Supplier":
            # src uses tgt as a supplier => tgt supplies src => risk flows tgt -> src
            G.add_edge(tgt, src)   # ✓ FIXED (was: G.add_edge(src, tgt))
 
        elif rel == "Customer":
            # src is a customer of tgt => tgt supplies src => risk flows tgt -> src
            G.add_edge(tgt, src)   # ✓ CORRECT (unchanged)
 
        else:
            # Fallback: treat as downstream
            G.add_edge(tgt, src)
 
    return G
 
 
# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import io
 
    # Minimal test
    test_csv = """source,target,relationship
Nvidia,TSMC,Supplier
Microsoft,Nvidia,Customer
Apple,TSMC,Supplier"""
 
    df = pd.read_csv(io.StringIO(test_csv))
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, tgt, rel = row["source"], row["target"], row["relationship"]
        G.add_edge(tgt, src)
 
    print("Edges (should all be: supplier/upstream -> customer/downstream):")
    for u, v in G.edges():
        print(f"  {u} --> {v}")
 
    print("\nExpected:")
    print("  TSMC --> Nvidia")
    print("  Nvidia --> Microsoft")
    print("  TSMC --> Apple")
 
    # Downstream from TSMC
    print(f"\nDownstream of TSMC: {list(G.successors('TSMC'))}")
    print(f"Upstream of Nvidia: {list(G.predecessors('Nvidia'))}")
 

def bfs_propagate(
    G: nx.DiGraph,
    epicenters: List[str],
    intensity: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """BFS propagation over real graph edges.

    Downstream (supplier shock → customers): decay 0.72 per hop
    Upstream   (demand shock → suppliers):   decay 0.42 per hop
    Random noise per edge if rng is provided (Monte Carlo mode).
    """
    if rng is None:
        rng = np.random.default_rng()

    scores = {n: 0.0 for n in G.nodes}
    for e in epicenters:
        if e in scores:
            scores[e] = float(intensity)

    DECAY_DOWN = 0.72
    DECAY_UP   = 0.42

    # ── downstream ──────────────────────────────
    queue, visited = list(epicenters), set(epicenters)
    while queue:
        node = queue.pop(0)
        node_risk = scores[node]
        if node_risk < 0.04:
            continue
        for customer in G.successors(node):
            prop = node_risk * DECAY_DOWN * rng.uniform(0.75, 1.15)
            prop = min(prop, 1.0)
            if prop > scores[customer]:
                scores[customer] = prop
                if customer not in visited:
                    visited.add(customer)
                    queue.append(customer)

    # ── upstream ────────────────────────────────
    queue, visited = list(epicenters), set(epicenters)
    while queue:
        node = queue.pop(0)
        node_risk = scores[node]
        if node_risk < 0.04:
            continue
        for supplier in G.predecessors(node):
            prop = node_risk * DECAY_UP * rng.uniform(0.6, 1.0)
            prop = min(prop, 1.0)
            if prop > scores[supplier]:
                scores[supplier] = prop
                if supplier not in visited:
                    visited.add(supplier)
                    queue.append(supplier)

    return scores


def monte_carlo(
    G: nx.DiGraph,
    epicenters: List[str],
    intensity: float,
    n: int = 500,
) -> Dict[str, Dict]:
    """Run n BFS iterations with perturbed intensity, return per-node statistics."""
    per_node: Dict[str, list] = {node: [] for node in G.nodes}
    rng = np.random.default_rng(42)

    for _ in range(n):
        perturbed = float(np.clip(intensity * rng.uniform(0.80, 1.20), 0, 1))
        scores = bfs_propagate(G, epicenters, perturbed, rng)
        for node, score in scores.items():
            per_node[node].append(score)

    stats = {}
    for node, samples in per_node.items():
        arr = np.array(samples)
        stats[node] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "p5":   float(np.percentile(arr, 5)),
            "p50":  float(np.percentile(arr, 50)),
            "p95":  float(np.percentile(arr, 95)),
        }
    return stats


# ── GNN inference (optional) ──────────────────────────────────────────────────

def gnn_predict(
    epicenters: List[str],
    intensity: float,
) -> Optional[Dict[str, float]]:
    """Run trained GNN if available. Returns None if not possible."""
    if not TORCH_AVAILABLE or state["gnn_model"] is None:
        return None
    try:
        model   = state["gnn_model"]
        data    = state["gnn_data"]
        c2i     = state["company_to_idx"]
        i2c     = state["idx_to_company"]
        device  = state["device"]

        x          = data.x.clone().to(device)
        edge_index = data.edge_index.to(device)

        shock = torch.zeros(x.shape[0], 1).to(device)
        for company in epicenters:
            if company in c2i:
                shock[c2i[company]] = intensity

        x_aug = torch.cat([x, shock], dim=1)

        with torch.no_grad():
            raw = model(x_aug, edge_index)

        return {i2c[i]: float(raw[i].cpu().item()) for i in i2c}
    except Exception as exc:
        print(f"[GNN] inference failed: {exc}")
        return None


# ── Risk categorisation ───────────────────────────────────────────────────────

def categorise(score: float) -> str:
    if score > 0.70: return "critical"
    if score > 0.50: return "high"
    if score > 0.30: return "moderate"
    return "low"


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    # 1. Load graph
    try:
        G = load_graph(CONNECTIONS_PATH)
        state["graph"] = G
        companies = list(G.nodes)
        state["company_to_idx"] = {c: i for i, c in enumerate(companies)}
        state["idx_to_company"] = {i: c for i, c in enumerate(companies)}
        print(f"[startup] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"[startup] Graph load failed: {e}")

    # 2. Try loading GNN weights
    if TORCH_AVAILABLE:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state["device"] = device

            data = torch.load(os.path.join(MODEL_DIR, "graph_data.pt"),    map_location=device)
            with open(os.path.join(MODEL_DIR, "company_mappings.pkl"), "rb") as f:
                mappings = pickle.load(f)

            model = SupplyChainGNN(
                input_dim=data.x.shape[1] + 1,
                hidden_dim=64, num_layers=3, dropout=0.2, heads=4,
            )
            ckpt = torch.load(os.path.join(MODEL_DIR, "supply_chain_gnn.pt"), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()

            state["gnn_model"]       = model
            state["gnn_data"]        = data
            state["company_to_idx"]  = mappings["company_to_idx"]
            state["idx_to_company"]  = mappings["idx_to_company"]
            print("[startup] GNN model loaded ✓")
        except FileNotFoundError:
            print("[startup] GNN weights not found — using BFS fallback")
        except Exception as e:
            print(f"[startup] GNN load error: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":       "healthy",
        "graph_loaded": state["graph"] is not None,
        "gnn_loaded":   state["gnn_model"] is not None,
        "nodes":        state["graph"].number_of_nodes() if state["graph"] else 0,
        "edges":        state["graph"].number_of_edges() if state["graph"] else 0,
        "timestamp":    datetime.now().isoformat(),
    }


@app.post("/api/predict", response_model=ScenarioResult)
def predict(scenario: ShockScenario):
    G = state["graph"]
    if G is None:
        raise HTTPException(503, "Graph not loaded")

    # Try GNN first, fall back to BFS
    gnn_scores = gnn_predict(scenario.affected_companies, scenario.intensity)
    if gnn_scores:
        scores        = gnn_scores
        inference_mode = "gnn"
    else:
        scores        = bfs_propagate(G, scenario.affected_companies, scenario.intensity)
        inference_mode = "bfs"

    # Optional Monte Carlo
    mc_stats: Optional[Dict] = None
    if scenario.monte_carlo_n > 0:
        mc_stats = monte_carlo(G, scenario.affected_companies, scenario.intensity, scenario.monte_carlo_n)

    # Build response
    predictions, counts = [], {"critical":0,"high":0,"moderate":0,"low":0}
    for company, score in sorted(scores.items(), key=lambda x: -x[1]):
        level = categorise(score)
        counts[level] += 1
        node = NodeRisk(
            company     = company,
            risk_score  = round(score, 4),
            risk_level  = level,
            is_epicenter= company in scenario.affected_companies,
        )
        if mc_stats and company in mc_stats:
            s = mc_stats[company]
            node.p5  = round(s["p5"],  4)
            node.p50 = round(s["p50"], 4)
            node.p95 = round(s["p95"], 4)
            node.std = round(s["std"],  4)
        predictions.append(node)

    return ScenarioResult(
        scenario_name      = scenario.scenario_name or "Custom",
        affected_companies = scenario.affected_companies,
        intensity          = scenario.intensity,
        inference_mode     = inference_mode,
        timestamp          = datetime.now().isoformat(),
        predictions        = predictions,
        summary            = counts,
    )


@app.get("/api/scenarios")
def list_scenarios():
    return {"scenarios": [
        {"name":"Taiwan Earthquake",       "affected":["TSMC"],                         "intensity":0.90,"category":"Natural Disaster"},
        {"name":"Lithium Supply Shock",    "affected":["Panasonic","LG","CATL","Samsung"],"intensity":0.80,"category":"Resource"},
        {"name":"AI Chip Shortage",        "affected":["Nvidia"],                        "intensity":0.70,"category":"Demand Shock"},
        {"name":"Rare Earth Export Ban",   "affected":["ASML","Samsung","SK_Hynix"],     "intensity":0.85,"category":"Geopolitical"},
        {"name":"Energy Supply Crisis",    "affected":["Samsung","TSMC","Intel","ASML"], "intensity":0.75,"category":"Energy"},
        {"name":"Port Disruption",         "affected":["Foxconn","Samsung","TSMC","Sony"],"intensity":0.65,"category":"Logistics"},
    ]}


@app.get("/api/graph")
def graph_info():
    G = state["graph"]
    if G is None:
        raise HTTPException(503, "Graph not loaded")
    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)
    return {
        "nodes": [
            {
                "company":      n,
                "in_degree":    G.in_degree(n),
                "out_degree":   G.out_degree(n),
                "pagerank":     round(pr[n], 5),
                "betweenness":  round(bc[n], 5),
            }
            for n in sorted(G.nodes, key=lambda x: -pr[x])
        ],
        "edge_count": G.number_of_edges(),
    }


@app.get("/api/companies")
def companies():
    G = state["graph"]
    if G is None:
        raise HTTPException(503, "Graph not loaded")
    return {"companies": sorted(G.nodes), "count": G.number_of_nodes()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)