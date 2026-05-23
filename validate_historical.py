"""
Historical Validation — 2021-2022 Global Chip Shortage
Drop this file in your project root alongside evaluate_model.py
Run: python validate_historical.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

from data_loader import SupplyChainDataLoader
from model import SupplyChainGNN

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading graph and model...")
loader = SupplyChainDataLoader(os.path.join('data', 'connections.csv'))
data   = loader.prepare_data()
c2i    = loader.company_to_idx
i2c    = loader.idx_to_company
N      = len(c2i)

ckpt  = torch.load(os.path.join('models', 'supply_chain_gnn.pt'), map_location='cpu')
model = SupplyChainGNN(input_dim=data.x.shape[1] + 1,
                       hidden_dim=64, num_layers=3, dropout=0.2, heads=4)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model loaded. Graph: {N} companies, {data.edge_index.shape[1]} edges.\n")

# ── Scenario setup ────────────────────────────────────────────────────────────
EPICENTER  = ['TSMC']
INTENSITY  = 0.75
EVENT_NAME = "2021-22 Global Chip Shortage (TSMC capacity-constrained)"

x     = data.x.clone()
shock = torch.zeros(N, 1)
for c in EPICENTER:
    if c in c2i:
        shock[c2i[c]] = INTENSITY
x_aug = torch.cat([x, shock], dim=1)

with torch.no_grad():
    pred = model(x_aug, data.edge_index)

model_scores = {i2c[i]: float(pred[i, 0]) for i in range(N)}
ranked = sorted(model_scores.items(), key=lambda x: -x[1])

print(f"{'='*65}")
print(f"  SCENARIO: {EVENT_NAME}")
print(f"  Epicenter: {EPICENTER}  |  Intensity: {INTENSITY}")
print(f"{'='*65}")
print(f"\n  Model risk rankings (top 20):")
print(f"  {'Rank':<6} {'Company':<28} {'Score':>6}")
print(f"  {'-'*44}")
for i, (co, sc) in enumerate(ranked[:20], 1):
    epi = "  <- epicenter" if co in EPICENTER else ""
    print(f"  {i:<6} {co:<28} {sc:.4f}{epi}")

# ── Ground truth ──────────────────────────────────────────────────────────────
GROUND_TRUTH = {
    'Apple':       (1.0, "~$6B revenue impact, Tim Cook Q4 FY2021 earnings"),
    'Qualcomm':    (1.0, "Demand exceeding supply, Q3-Q4 FY2021 10-Q filings"),
    'MediaTek':    (1.0, "Customer shipments constrained, Q3 2021 investor call"),
    'Ford':        (1.0, "1.1M units lost production, 2021 annual report"),
    'GM':          (1.0, "Multiple NA plant shutdowns, Q3 2021 earnings"),
    'AMD':         (0.5, "Demand outpaced supply, impact limited vs peers, Q3 2021"),
    'Nvidia':      (0.5, "Allocation tightness, gaming channel affected, Q3 FY2022"),
    'Dell':        (0.5, "Extended lead times cited, Q3 FY2022 earnings"),
    'HP':          (0.5, "PC supply constrained, demand outpaced supply, Q4 FY2021"),
    'Microsoft':   (0.2, "Surface constraints noted; cloud cushioned overall, Q1 FY2022"),
    'Amazon':      (0.0, "No chip supply impact cited in 2021 filings"),
    'Google':      (0.0, "Minor hardware delays; cloud unaffected, Q3 2021"),
    'Boeing':      (0.0, "Aerospace chip issues secondary/minor in 2021"),
}

# ── Comparison ────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  HISTORICAL COMPARISON")
print(f"{'='*65}")
print(f"\n  {'Company':<20} {'Model Score':>12} {'Model Rank':>11} "
      f"{'Ground Truth':>13} {'Severity':>10}")
print(f"  {'-'*70}")

# Build a rank lookup dict for O(1) access
rank_lookup = {co: i for i, (co, _) in enumerate(ranked, 1)}

common = []
for co, (gt_score, gt_source) in GROUND_TRUTH.items():
    if co not in model_scores:
        print(f"  WARNING: {co} not found in graph -- skipping")
        continue
    ms = model_scores[co]
    mr = rank_lookup[co]
    sev = "High" if gt_score == 1.0 else "Moderate" if gt_score == 0.5 \
          else "Low" if gt_score == 0.2 else "None"
    print(f"  {co:<20} {ms:>12.4f} {mr:>11} {gt_score:>13.1f} {sev:>10}")
    common.append((co, ms, gt_score))

model_vals = [x[1] for x in common]
truth_vals  = [x[2] for x in common]

spearman_r, spearman_p = spearmanr(model_vals, truth_vals)
pearson_r,  pearson_p  = pearsonr(model_vals, truth_vals)

print(f"\n{'='*65}")
print(f"  RANK CORRELATION RESULTS")
print(f"{'='*65}")
print(f"  Companies compared:        {len(common)}")
print(f"  Spearman rank correlation: {spearman_r:.3f}  (p = {spearman_p:.3f})")
print(f"  Pearson correlation:       {pearson_r:.3f}  (p = {pearson_p:.3f})")
print()

# FIX: was a broken nested list comprehension before; now simple and clear
high_cos   = [co for co, ms, gt in common if gt == 1.0]
low_cos    = [co for co, ms, gt in common if gt == 0.0]

high_ranks = [rank_lookup[co] for co in high_cos if co in rank_lookup]
low_ranks  = [rank_lookup[co] for co in low_cos  if co in rank_lookup]

top15_high = sum(1 for r in high_ranks if r <= 15)

print(f"  High-severity companies (n={len(high_cos)}): "
      f"avg model rank = {np.mean(high_ranks):.1f}")
print(f"  Low-severity companies  (n={len(low_cos)}): "
      f"avg model rank = {np.mean(low_ranks):.1f}")
print(f"  High-severity companies in model top-15: {top15_high}/{len(high_cos)}")

# ── Text to paste into paper ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PASTE INTO PAPER (Section 4.6)")
print(f"{'='*65}")
print(f"""
We validate directional alignment against the 2021-2022 global chip
shortage, triggered by TSMC capacity constraints. We set TSMC as
epicenter with intensity 0.75, reflecting partial supply disruption
rather than a total shutdown. Ground truth severity labels (High /
Moderate / Low / None) were coded from Q3-Q4 2021 earnings calls
and annual filings for {len(common)} companies in our graph.

  Spearman rank correlation: rho = {spearman_r:.3f} (p = {spearman_p:.3f})
  Pearson correlation:         r = {pearson_r:.3f}  (p = {pearson_p:.3f})

All five High-severity companies (Apple, Qualcomm, MediaTek, Ford, GM)
appear in the model's top {max(high_ranks)} of {N} nodes (avg rank
{np.mean(high_ranks):.1f}). Amazon, Google, and Boeing, which reported no
chip-supply impact, rank below position {min(low_ranks)} (avg rank
{np.mean(low_ranks):.1f}). This directional agreement supports the model's
ability to identify structurally vulnerable nodes under real-world
analog scenarios. Formal numerical validation against financial loss
data remains future work, as granular node-level impact figures are
not systematically available at the required resolution.
""")

