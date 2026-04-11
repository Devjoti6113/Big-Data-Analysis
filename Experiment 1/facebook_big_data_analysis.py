#!/usr/bin/env python3
"""
==========================================================================
Big Data Analysis Project - Facebook Social Network Analysis
IITKGP - Big Data Analysis Class
==========================================================================

Dataset: MUSAE Facebook Page-Page Network
  - 22,470 nodes (Facebook verified pages)
  - 171,002 undirected edges (mutual likes between pages)

Pipeline:
  Step 1: Load data & compute network statistics
  Step 2: Visualize the network structure (multiple views)
  Step 3: Compute K-distance induced neighbourhood subgraphs
  Step 4: Extract 20-dimensional degree-based feature vectors
  Step 5: PCA: Determine components for 90%, 80%, 60% variance
==========================================================================
"""

import os
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import Counter, deque

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_PATH = 'MUSAE-master/input/edges/facebook_edges.csv'
OUTPUT_DIR = 'output_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def separator(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: Load Dataset & Build Undirected Graph                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
separator("STEP 1: Loading Facebook Dataset & Building Graph")

edges_df = pd.read_csv(DATA_PATH)
print(f"  CSV shape: {edges_df.shape}")
print(f"  Columns:   {list(edges_df.columns)}")
print(f"  Sample rows:")
print(edges_df.head().to_string(index=False))

# Build an undirected graph (Step 3 of the project requirement)
G = nx.from_pandas_edgelist(edges_df, 'id_1', 'id_2')

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  Graph Statistics                        │")
print(f"  ├─────────────────────────────────────────┤")
print(f"  │  Nodes (N):             {num_nodes:>14,}  │")
print(f"  │  Edges:                 {num_edges:>14,}  │")
print(f"  │  Density:               {nx.density(G):>14.6f}  │")

# Connected components
components = sorted(nx.connected_components(G), key=len, reverse=True)
print(f"  │  Connected Components:  {len(components):>14}  │")
print(f"  │  Largest Component:     {len(components[0]):>14,}  │")

# Degree statistics
degrees_dict = dict(G.degree())
degree_values = np.array(list(degrees_dict.values()))
print(f"  │  Mean Degree:           {np.mean(degree_values):>14.2f}  │")
print(f"  │  Median Degree:         {np.median(degree_values):>14.1f}  │")
print(f"  │  Max Degree:            {np.max(degree_values):>14}  │")
print(f"  │  Min Degree:            {np.min(degree_values):>14}  │")
print(f"  │  Std Degree:            {np.std(degree_values):>14.2f}  │")
print(f"  └─────────────────────────────────────────┘")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: Network Visualization                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
separator("STEP 2: Network Visualization")

# ── 2a. Degree Distribution ────────────────────────────────────────────
print("  [2a] Plotting degree distribution...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Facebook Page-Page Network — Degree Distribution', fontsize=18, fontweight='bold', y=1.02)

# Linear histogram
axes[0].hist(degree_values, bins=80, color='#3498db', edgecolor='#2c3e50', alpha=0.85, linewidth=0.5)
axes[0].set_xlabel('Degree', fontsize=14)
axes[0].set_ylabel('Frequency', fontsize=14)
axes[0].set_title('Linear Scale', fontsize=15, fontweight='bold')
axes[0].axvline(np.mean(degree_values), color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(degree_values):.1f}')
axes[0].axvline(np.median(degree_values), color='#2ecc71', linestyle='--', linewidth=2,
                label=f'Median = {np.median(degree_values):.0f}')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Log-log scatter (power-law check)
degree_count = Counter(degree_values)
deg_vals, counts = zip(*sorted(degree_count.items()))
axes[1].scatter(deg_vals, counts, c='#e74c3c', s=18, alpha=0.7, edgecolors='none', zorder=5)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel('Degree (log)', fontsize=14)
axes[1].set_ylabel('Count (log)', fontsize=14)
axes[1].set_title('Log-Log Scale (Power-Law Check)', fontsize=15, fontweight='bold')
axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_degree_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("       ✓ Saved: 01_degree_distribution.png")


# ── 2b. Network Graph Visualization ───────────────────────────────────
SAMPLE_SIZE = 2500
print(f"\n  [2b] Network visualization ({SAMPLE_SIZE} nodes via BFS from hub)...")

# BFS from the highest-degree node to get a connected, representative subgraph
hub_node = max(degrees_dict, key=degrees_dict.get)
print(f"       Hub node: {hub_node} (degree={degrees_dict[hub_node]})")

visited = set()
queue = deque([hub_node])
visited.add(hub_node)
while len(visited) < SAMPLE_SIZE and queue:
    node = queue.popleft()
    for neighbor in G.neighbors(node):
        if neighbor not in visited and len(visited) < SAMPLE_SIZE:
            visited.add(neighbor)
            queue.append(neighbor)

G_viz = G.subgraph(visited).copy()
print(f"       Subgraph: {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges")

print("       Computing spring layout (may take 30-60s)...")
t0 = time.time()
pos = nx.spring_layout(G_viz, k=0.06, iterations=50, seed=RANDOM_SEED)
print(f"       Layout computed in {time.time()-t0:.1f}s")

# Node properties
node_deg = np.array([G_viz.degree(n) for n in G_viz.nodes()])
max_deg_viz = node_deg.max() if len(node_deg) > 0 else 1
node_sizes = 5 + 80 * (node_deg / max_deg_viz) ** 1.5

# Create figure with dark background
fig, ax = plt.subplots(1, 1, figsize=(28, 28), facecolor='#0a0a1a')
ax.set_facecolor('#0a0a1a')

# Draw edges
nx.draw_networkx_edges(G_viz, pos, ax=ax, width=0.12, alpha=0.06, edge_color='#4ecdc4')

# Draw nodes colored by degree
norm = Normalize(vmin=node_deg.min(), vmax=np.percentile(node_deg, 95))
node_colors = cm.plasma(norm(node_deg))
nx.draw_networkx_nodes(G_viz, pos, ax=ax,
                       node_size=node_sizes,
                       node_color=node_colors,
                       alpha=0.9,
                       linewidths=0)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.01, aspect=30)
cbar.set_label('Node Degree', fontsize=16, color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax.set_title(f'Facebook Social Network — {SAMPLE_SIZE} Nodes (BFS from Hub)',
             fontsize=26, fontweight='bold', color='white', pad=20)
ax.axis('off')

plt.savefig(f'{OUTPUT_DIR}/02_network_graph.png', dpi=180, bbox_inches='tight',
            facecolor='#0a0a1a', edgecolor='none')
plt.close()
print("       ✓ Saved: 02_network_graph.png")


# ── 2c. Community-Colored Visualization ───────────────────────────────
print(f"\n  [2c] Community detection & visualization...")

# Use Louvain-like greedy modularity for community detection
from networkx.algorithms.community import greedy_modularity_communities

communities = list(greedy_modularity_communities(G_viz, resolution=1.2))
print(f"       Detected {len(communities)} communities")

# Assign community labels
node_community = {}
for idx, comm in enumerate(communities):
    for node in comm:
        node_community[node] = idx

comm_labels = [node_community.get(n, 0) for n in G_viz.nodes()]

# Color by community
num_comm = len(communities)
cmap_comm = plt.cm.get_cmap('tab20', min(num_comm, 20))
comm_colors = [cmap_comm(c % 20) for c in comm_labels]

fig, ax = plt.subplots(1, 1, figsize=(28, 28), facecolor='#0d1117')
ax.set_facecolor('#0d1117')

nx.draw_networkx_edges(G_viz, pos, ax=ax, width=0.12, alpha=0.05, edge_color='#555555')
nx.draw_networkx_nodes(G_viz, pos, ax=ax,
                       node_size=node_sizes,
                       node_color=comm_colors,
                       alpha=0.9,
                       linewidths=0)

ax.set_title(f'Facebook Network — Community Structure ({len(communities)} Communities)',
             fontsize=26, fontweight='bold', color='white', pad=20)
ax.axis('off')

plt.savefig(f'{OUTPUT_DIR}/03_community_graph.png', dpi=180, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print("       ✓ Saved: 03_community_graph.png")


# ── 2d. Top-20 Hub Nodes with Labels ─────────────────────────────────
print(f"\n  [2d] Hub nodes ego visualization...")

top_hubs = sorted(degrees_dict, key=degrees_dict.get, reverse=True)[:20]
print(f"       Top-20 hubs by degree: {[(h, degrees_dict[h]) for h in top_hubs[:5]]}...")

# Get 1-hop ego network of top hubs
hub_ego_nodes = set(top_hubs)
for hub in top_hubs:
    for neighbor in G.neighbors(hub):
        hub_ego_nodes.add(neighbor)
        if len(hub_ego_nodes) > 3000:
            break
    if len(hub_ego_nodes) > 3000:
        break

G_ego = G.subgraph(hub_ego_nodes).copy()
pos_ego = nx.spring_layout(G_ego, k=0.1, iterations=40, seed=RANDOM_SEED)

fig, ax = plt.subplots(figsize=(20, 20), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Draw all nodes small
ego_deg = np.array([G_ego.degree(n) for n in G_ego.nodes()])
nx.draw_networkx_edges(G_ego, pos_ego, ax=ax, width=0.1, alpha=0.04, edge_color='#16213e')

is_hub = np.array([n in top_hubs for n in G_ego.nodes()])
non_hub_nodes = [n for n in G_ego.nodes() if n not in top_hubs]
hub_nodes_in_ego = [n for n in G_ego.nodes() if n in top_hubs]

# Non-hub nodes
nx.draw_networkx_nodes(G_ego, pos_ego, nodelist=non_hub_nodes, ax=ax,
                       node_size=3, node_color='#e2e2e2', alpha=0.3, linewidths=0)

# Hub nodes — large and glowing
hub_sizes = [150 + degrees_dict[n] * 2 for n in hub_nodes_in_ego]
nx.draw_networkx_nodes(G_ego, pos_ego, nodelist=hub_nodes_in_ego, ax=ax,
                       node_size=hub_sizes, node_color='#ff6b6b', alpha=0.95,
                       linewidths=1.5, edgecolors='#feca57')

# Labels for hubs
hub_labels = {n: f"{n}\n(d={degrees_dict[n]})" for n in hub_nodes_in_ego}
nx.draw_networkx_labels(G_ego, pos_ego, labels=hub_labels, ax=ax,
                        font_size=7, font_color='#feca57', font_weight='bold')

ax.set_title('Top-20 Hub Nodes in the Facebook Network',
             fontsize=22, fontweight='bold', color='white', pad=20)
ax.axis('off')

plt.savefig(f'{OUTPUT_DIR}/04_hub_nodes.png', dpi=180, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("       ✓ Saved: 04_hub_nodes.png")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 & 4: K-Distance Neighborhoods + Feature Extraction        ║
# ╚══════════════════════════════════════════════════════════════════════╝
separator("STEP 3 & 4: K-Distance Neighbourhoods & Feature Extraction")

# ── Analyze neighbourhood expansion to choose K wisely ─────────────────
print("  Analyzing neighbourhood expansion for different K values...")
print("  (Sampling 200 random nodes to estimate)")

test_nodes = random.sample(list(G.nodes()), min(200, num_nodes))
print(f"\n  {'K':>4s} | {'Avg Size':>10s} | {'Median':>8s} | {'Max':>8s} | {'% of Graph':>11s}")
print(f"  {'─'*4}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*11}")

for k_test in [1, 2, 3, 4, 5]:
    sizes = []
    for node in test_nodes:
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=k_test)
        sizes.append(len(neighbors))
    avg_s = np.mean(sizes)
    med_s = np.median(sizes)
    max_s = np.max(sizes)
    pct = avg_s / num_nodes * 100
    print(f"  {k_test:>4d} | {avg_s:>10.0f} | {med_s:>8.0f} | {max_s:>8d} | {pct:>10.1f}%")

# ── Choose K ───────────────────────────────────────────────────────────
K = 2
print(f"\n  ★ Selected K = {K}")
print(f"    Rationale: With K=2, neighbourhood subgraphs capture the 2-hop")
print(f"    local structure around each node. Higher K values (≥4) encompass")
print(f"    nearly the entire graph due to the small-world property, making")
print(f"    all feature vectors essentially identical — reducing discriminative power.")
print(f"    K=2 provides meaningful structural variation across the {num_nodes} nodes.\n")

# ── Feature Matrix Construction ────────────────────────────────────────
FEATURE_DIM = 20
N = num_nodes
nodes_list = sorted(G.nodes())

print(f"  Building {FEATURE_DIM}-dimensional feature vectors for all {N} nodes...")
print(f"  Feature = top-{FEATURE_DIM} degree values in K={K} neighbourhood subgraph\n")

feature_matrix = np.zeros((N, FEATURE_DIM))
neighbourhood_sizes = []

t0 = time.time()
for i, node in enumerate(nodes_list):
    # Get K-distance neighbourhood via BFS
    neighbors = nx.single_source_shortest_path_length(G, node, cutoff=K)
    neighbourhood_nodes = list(neighbors.keys())
    neighbourhood_sizes.append(len(neighbourhood_nodes))

    # Induced subgraph
    subgraph = G.subgraph(neighbourhood_nodes)

    # Get all degrees within this subgraph
    sub_degrees = sorted([d for _, d in subgraph.degree()], reverse=True)

    # Take top-20 highest degrees
    top_k = sub_degrees[:FEATURE_DIM]

    # Pad with zeros if subgraph has fewer than 20 nodes
    while len(top_k) < FEATURE_DIM:
        top_k.append(0)

    feature_matrix[i] = top_k

    if (i + 1) % 2000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (N - i - 1) / rate
        print(f"    [{i+1:6d}/{N}]  elapsed: {elapsed:6.1f}s  |  rate: {rate:.0f} nodes/s  |  ETA: {remaining:.0f}s")

total_time = time.time() - t0
print(f"\n  ✓ Feature extraction complete in {total_time:.1f}s")
print(f"    Feature matrix shape: {feature_matrix.shape}")
print(f"    Avg neighbourhood size: {np.mean(neighbourhood_sizes):.1f}")
print(f"    Min neighbourhood size: {np.min(neighbourhood_sizes)}")
print(f"    Max neighbourhood size: {np.max(neighbourhood_sizes)}")

# Save feature matrix
csv_path = f'{OUTPUT_DIR}/feature_matrix_N{N}_K{K}_D{FEATURE_DIM}.csv'
np.savetxt(csv_path, feature_matrix, delimiter=',', fmt='%.0f',
           header=','.join([f'deg_rank_{j+1}' for j in range(FEATURE_DIM)]), comments='')
print(f"    Saved: {csv_path}")

# ── Visualize Feature Matrix Heatmap ───────────────────────────────────
print("\n  Plotting feature matrix heatmap...")

# Sort rows by first feature (highest degree in neighbourhood) for a cleaner view
sort_idx = np.argsort(-feature_matrix[:, 0])
sorted_fm = feature_matrix[sort_idx]

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(sorted_fm, aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_xlabel(f'Feature Dimension (Top-{FEATURE_DIM} Degrees)', fontsize=14)
ax.set_ylabel(f'Nodes (sorted by 1st feature, N={N})', fontsize=14)
ax.set_title(f'Feature Matrix Heatmap — {N} Nodes × {FEATURE_DIM} Features (K={K})',
             fontsize=16, fontweight='bold')
ax.set_xticks(range(FEATURE_DIM))
ax.set_xticklabels([f'D{j+1}' for j in range(FEATURE_DIM)], fontsize=9)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Degree Value', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_feature_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("       ✓ Saved: 05_feature_heatmap.png")

# ── Neighbourhood Size Distribution ───────────────────────────────────
print("  Plotting neighbourhood size distribution...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(neighbourhood_sizes, bins=60, color='#2ecc71', edgecolor='#27ae60', alpha=0.85, linewidth=0.5)
ax.set_xlabel(f'{K}-Distance Neighbourhood Size', fontsize=14)
ax.set_ylabel('Number of Nodes', fontsize=14)
ax.set_title(f'Distribution of K={K} Neighbourhood Sizes (N={N} nodes)',
             fontsize=16, fontweight='bold')
ax.axvline(np.mean(neighbourhood_sizes), color='#e74c3c', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(neighbourhood_sizes):.0f}')
ax.axvline(np.median(neighbourhood_sizes), color='#3498db', linestyle='--', linewidth=2,
           label=f'Median: {np.median(neighbourhood_sizes):.0f}')
ax.legend(fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_neighbourhood_sizes.png', dpi=200, bbox_inches='tight')
plt.close()
print("       ✓ Saved: 06_neighbourhood_sizes.png")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: Principal Component Analysis (PCA)                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
separator("STEP 5: Principal Component Analysis (from scratch)")

# ── 5a. Standardize the Feature Matrix ─────────────────────────────────
print("  [5a] Standardizing features (zero mean, unit variance)...")

X = feature_matrix.copy()
mean_vec = np.mean(X, axis=0)
std_vec = np.std(X, axis=0)
std_vec[std_vec == 0] = 1.0  # avoid division by zero

X_scaled = (X - mean_vec) / std_vec

print(f"       Shape: {X_scaled.shape}")
print(f"       Mean after scaling: {np.mean(X_scaled, axis=0).round(6)[:5]}... (should be ~0)")
print(f"       Std after scaling:  {np.std(X_scaled, axis=0).round(4)[:5]}...  (should be ~1)")

# ── 5b. Covariance Matrix ─────────────────────────────────────────────
print("\n  [5b] Computing covariance matrix...")

cov_matrix = np.cov(X_scaled, rowvar=False)  # (20, 20)
print(f"       Covariance matrix shape: {cov_matrix.shape}")

# ── 5c. Eigendecomposition ─────────────────────────────────────────────
print("\n  [5c] Eigendecomposition of covariance matrix...")

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues (and corresponding eigenvectors) in DESCENDING order
idx_sorted = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx_sorted]
eigenvectors = eigenvectors[:, idx_sorted]

# Explained variance
total_variance = np.sum(eigenvalues)
explained_var_ratio = eigenvalues / total_variance
cumulative_var = np.cumsum(explained_var_ratio)

print(f"\n  ┌────────┬──────────────┬────────────────┬──────────────────┐")
print(f"  │   PC   │  Eigenvalue  │ Var. Explained │ Cumulative Var.  │")
print(f"  ├────────┼──────────────┼────────────────┼──────────────────┤")
for i in range(FEATURE_DIM):
    marker = ""
    if cumulative_var[i] >= 0.60 and (i == 0 or cumulative_var[i-1] < 0.60):
        marker = " ◄── 60%"
    elif cumulative_var[i] >= 0.80 and (i == 0 or cumulative_var[i-1] < 0.80):
        marker = " ◄── 80%"
    elif cumulative_var[i] >= 0.90 and (i == 0 or cumulative_var[i-1] < 0.90):
        marker = " ◄── 90%"
    print(f"  │  PC{i+1:<3d} │ {eigenvalues[i]:>12.6f} │ {explained_var_ratio[i]*100:>12.4f}%  │ {cumulative_var[i]*100:>14.4f}%  │{marker}")
print(f"  └────────┴──────────────┴────────────────┴──────────────────┘")

# ── 5d. Answer: How many PCs for 60%, 80%, 90% ────────────────────────
print(f"\n  ★ RESULTS: Number of Principal Components Required")
print(f"  ┌─────────────────────────┬──────────────────────────┐")
print(f"  │  Variance Threshold     │  # of PCs Required       │")
print(f"  ├─────────────────────────┼──────────────────────────┤")

thresholds = [0.60, 0.80, 0.90]
pcs_needed = {}
for thresh in thresholds:
    n_pc = int(np.searchsorted(cumulative_var, thresh) + 1)
    pcs_needed[thresh] = n_pc
    actual_var = cumulative_var[n_pc - 1] * 100
    print(f"  │  {thresh*100:5.0f}% variance         │  {n_pc:3d} PCs ({actual_var:.2f}% actual)  │")

print(f"  └─────────────────────────┴──────────────────────────┘")

# ── 5e. Scree Plot + Cumulative Variance Plot ─────────────────────────
print("\n  Plotting PCA variance analysis...")

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle(f'PCA Variance Analysis — Facebook Network (N={N}, K={K}, D={FEATURE_DIM})',
             fontsize=18, fontweight='bold', y=1.02)

# (i) Individual Explained Variance (Bar)
colors_bar = ['#e74c3c' if i < pcs_needed[0.90] else '#95a5a6' for i in range(FEATURE_DIM)]
axes[0].bar(range(1, FEATURE_DIM + 1), explained_var_ratio * 100,
            color=colors_bar, edgecolor='#2c3e50', linewidth=0.5, alpha=0.9)
axes[0].set_xlabel('Principal Component', fontsize=13)
axes[0].set_ylabel('Explained Variance (%)', fontsize=13)
axes[0].set_title('Individual Explained Variance', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(1, FEATURE_DIM + 1))
axes[0].grid(True, alpha=0.3, axis='y')

# (ii) Cumulative Explained Variance
axes[1].plot(range(1, FEATURE_DIM + 1), cumulative_var * 100, 'o-',
             color='#2c3e50', linewidth=2.5, markersize=8, markerfacecolor='#3498db',
             markeredgecolor='#2c3e50', markeredgewidth=1.5, zorder=5)

# Threshold lines
for thresh, color, ls in [(0.90, '#e74c3c', '--'), (0.80, '#f39c12', '--'), (0.60, '#2ecc71', '--')]:
    n_pc = pcs_needed[thresh]
    axes[1].axhline(y=thresh * 100, color=color, linestyle=ls, linewidth=1.5, alpha=0.7)
    axes[1].axvline(x=n_pc, color=color, linestyle=':', linewidth=1.2, alpha=0.5)
    axes[1].annotate(f'{thresh*100:.0f}% → {n_pc} PCs',
                     xy=(n_pc, thresh * 100), xytext=(n_pc + 0.8, thresh * 100 - 3),
                     fontsize=11, fontweight='bold', color=color,
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

axes[1].set_xlabel('Number of Principal Components', fontsize=13)
axes[1].set_ylabel('Cumulative Variance Explained (%)', fontsize=13)
axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(1, FEATURE_DIM + 1))
axes[1].set_ylim(0, 105)
axes[1].grid(True, alpha=0.3)

# (iii) Scree Plot (Eigenvalues)
axes[2].plot(range(1, FEATURE_DIM + 1), eigenvalues, 's-',
             color='#8e44ad', linewidth=2, markersize=8, markerfacecolor='#9b59b6',
             markeredgecolor='#6c3483', markeredgewidth=1.5)
axes[2].set_xlabel('Principal Component', fontsize=13)
axes[2].set_ylabel('Eigenvalue', fontsize=13)
axes[2].set_title('Scree Plot', fontsize=14, fontweight='bold')
axes[2].set_xticks(range(1, FEATURE_DIM + 1))
axes[2].grid(True, alpha=0.3)

# Kaiser criterion (eigenvalue = 1)
axes[2].axhline(y=1.0, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7,
                label='Kaiser Criterion (λ=1)')
axes[2].legend(fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_pca_variance_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("       ✓ Saved: 07_pca_variance_analysis.png")

# ── 5f. 2D PCA Projection ─────────────────────────────────────────────
print("\n  Computing 2D and 3D PCA projections...")

# Project data onto principal components
X_pca = X_scaled @ eigenvectors  # (N, 20)

# 2D scatter (PC1 vs PC2)
fig, ax = plt.subplots(figsize=(14, 10))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=degree_values, cmap='plasma', s=5, alpha=0.5, edgecolors='none')
ax.set_xlabel(f'PC1 ({explained_var_ratio[0]*100:.2f}% variance)', fontsize=14)
ax.set_ylabel(f'PC2 ({explained_var_ratio[1]*100:.2f}% variance)', fontsize=14)
ax.set_title(f'2D PCA Projection — Facebook Network Nodes (colored by degree)',
             fontsize=16, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
cbar.set_label('Original Node Degree', fontsize=13)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_pca_2d_projection.png', dpi=200, bbox_inches='tight')
plt.close()
print("       ✓ Saved: 08_pca_2d_projection.png")

# 3D scatter (PC1, PC2, PC3)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                c=degree_values, cmap='plasma', s=3, alpha=0.4, edgecolors='none')
ax.set_xlabel(f'PC1 ({explained_var_ratio[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_var_ratio[1]*100:.1f}%)', fontsize=12)
ax.set_zlabel(f'PC3 ({explained_var_ratio[2]*100:.1f}%)', fontsize=12)
ax.set_title(f'3D PCA Projection — Facebook Network\n'
             f'(Top 3 PCs capture {cumulative_var[2]*100:.1f}% variance)',
             fontsize=16, fontweight='bold')
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('Node Degree', fontsize=12)
plt.savefig(f'{OUTPUT_DIR}/09_pca_3d_projection.png', dpi=200, bbox_inches='tight')
plt.close()
print("       ✓ Saved: 09_pca_3d_projection.png")

# ── 5g. Feature Correlation Heatmap ───────────────────────────────────
print("  Plotting feature correlation matrix...")

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cov_matrix, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
ax.set_xticks(range(FEATURE_DIM))
ax.set_yticks(range(FEATURE_DIM))
ax.set_xticklabels([f'D{j+1}' for j in range(FEATURE_DIM)], fontsize=9)
ax.set_yticklabels([f'D{j+1}' for j in range(FEATURE_DIM)], fontsize=9)
ax.set_title('Covariance Matrix of Standardized Features', fontsize=16, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Covariance', fontsize=13)

# Add text annotations for values
for i_r in range(FEATURE_DIM):
    for j_c in range(FEATURE_DIM):
        val = cov_matrix[i_r, j_c]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j_c, i_r, f'{val:.2f}', ha='center', va='center',
                fontsize=6, color=color)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_covariance_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("       ✓ Saved: 10_covariance_matrix.png")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
separator("FINAL SUMMARY")

print(f"  Dataset:     Facebook Page-Page Network (MUSAE)")
print(f"  Nodes (N):   {num_nodes:,}")
print(f"  Edges:       {num_edges:,}")
print(f"  K-distance:  K = {K}")
print(f"  Features:    {FEATURE_DIM}-dimensional (top-{FEATURE_DIM} degrees in K-neighbourhood)")
print(f"  PCA (from scratch via eigendecomposition of covariance matrix):")
print(f"")
print(f"  ┌──────────────────────────────────────────────────────┐")
print(f"  │           PCA VARIANCE CAPTURE RESULTS               │")
print(f"  ├──────────────────────────────────────────────────────┤")
for thresh in thresholds:
    n_pc = pcs_needed[thresh]
    actual_var = cumulative_var[n_pc - 1] * 100
    print(f"  │   {thresh*100:.0f}% variance  →  {n_pc:2d} principal components ({actual_var:.2f}%)  │")
print(f"  └──────────────────────────────────────────────────────┘")
print(f"")
print(f"  Output plots saved to: {os.path.abspath(OUTPUT_DIR)}/")
print(f"  Feature matrix saved:  {csv_path}")
print(f"\n  All {len(os.listdir(OUTPUT_DIR))} plots generated successfully!")
print(f"{'='*72}")
