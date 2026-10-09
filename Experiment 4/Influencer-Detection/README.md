# Social Network Influencer Detection

Which users of a trust network spread information furthest, and can cheap structural features
predict it without simulating? Data: the SNAP Epinions "who-trusts-whom" network
(75,879 users, 508,837 directed trust edges; 405,740 undirected edges; 1,624,481 triangles).

## Pipeline

1. **Graph**: load the directed edge list, keep in/out degree and reciprocity, build the undirected
   simple graph in CSR form. Giant component: 75,877 of 75,879 nodes.
2. **Topology** (compiled with numba, whole graph in ~10 s): k-core (bucket peeling, O(n+m)),
   k-truss (triangle support + edge peeling), triangles, clustering, two-hop reach, collective
   influence CI_2, PageRank, eigenvector centrality.
3. **Forman–Ricci curvature** per edge: plain F = 4 − deg(u) − deg(v) and augmented
   F# = F + 3·triangles(u,v); node mean, minimum, spread and share of negatively curved edges.
4. **20-D feature table** for every node; **PCA** to study redundancy.
5. **Ground truth by Monte Carlo SIR** from 1,000 seed nodes (100 from each degree decile of the
   giant component), β = 1.5 × epidemic threshold (β_c = ⟨k⟩/(⟨k²⟩−⟨k⟩) = 0.0055), γ = 1,
   1,000 runs per seed. "Influencer" = top 10 % of seeds by mean outbreak size.
6. **Models**, every score out of fold (5-fold): single features, the degree–coreness–trussness
   ensemble, and all 20 features. ROC-AUC for influencer classification, R² for log spread.

All settings (β multiple, runs, sample, label threshold, folds) were fixed before any result was seen.

## Results

| Model (5-fold, out of fold) | ROC-AUC | Avg. precision | R² (log spread) | Spearman |
|---|---|---|---|---|
| degree alone | 0.9914 | 0.922 | 0.712 | 0.701 |
| coreness alone | 0.9943 | 0.945 | 0.741 | 0.707 |
| trussness alone | 0.9656 | 0.692 | 0.532 | 0.524 |
| collective influence CI_2 alone | 0.9973 | 0.975 | 0.596 | 0.748 |
| two-hop reach alone | 0.9920 | 0.929 | 0.803 | 0.918 |
| −(node Forman curvature) alone | 0.8431 | 0.240 | 0.473 | 0.798 |
| **degree–coreness–trussness ensemble** | **0.9955** (0.9952–0.9958 over 10 CV splits) | 0.965 | 0.751 | 0.718 |
| **all 20 features** | **0.9988** | 0.990 | **0.910** (0.908–0.911) | 0.917 |

* **Noise ceiling.** Two independent 500-run halves of the simulation agree only to Spearman 0.83
  (R² of log spread 0.78). The 1,000-run average used as ground truth has reliability ≈ 0.91, so an
  R² near 0.91 is at the ceiling of what this ground truth allows.
* **Baseline.** Degree alone already scores 0.991 (95 % bootstrap interval 0.987–0.995). The ensemble's
  gain is small in AUC and larger in average precision (0.922 → 0.965).
* **Robustness to the infection rate** (200 runs each): at 2 × β_c the ensemble keeps AUC 0.995; at the
  threshold itself (most outbreaks die at once) every method drops to about 0.93.
* **PCA.** On raw features one component explains 99.99999 % of the variance: heavy-tailed counts
  (CI_2, triangles) swamp everything ("variance saturation"). After a signed log and standardisation
  PC1 explains 58.7 % and 5 components reach 90 %. A logistic model on PC1 alone scores AUC 0.995.
* **Forman–Ricci filtering.** Keeping only nodes that touch a strongly negatively curved edge
  (bottom 30 % by minimum incident curvature) removes 70 % of the graph and keeps 100 % of the top
  spreaders. At tighter cuts a plain degree filter of the same size keeps more of them
  (20 % kept: 94 % vs 100 %), because on an unweighted graph plain Forman curvature is a function of
  endpoint degrees. Curvature adds information only through the triangle term.
* **Distances.** Mean shortest path 4.35, 90th percentile 6, longest seen 13: a small-world graph.

## Reproduce

```bash
pip install numpy scipy pandas scikit-learn numba matplotlib networkx pytest
python -m pytest -q tests          # k-core and k-truss checked against NetworkX, curvature and SIR limits
python scripts/run_pipeline.py     # ~6 min on 2 cores (SIR dominates); writes results/
python scripts/stability.py        # CV-split and bootstrap stability of the headline numbers
```
