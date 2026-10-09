#!/usr/bin/env python3
"""End-to-end pipeline. Run from the project root:  python scripts/run_pipeline.py
Writes results/tables/*.csv|json and results/figures/*.png. SIR results are cached in results/sir_cache.npz."""
import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from influence import features, graph, sir                           # noqa: E402
from influence.evaluate import oof, transform, unfitted               # noqa: E402

OUT = "results"; T = os.path.join(OUT, "tables"); F = os.path.join(OUT, "figures")
os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)
BETA_MULT, RUNS, N_SEEDS, TOP = 1.5, 1000, 1000, 0.10                 # fixed before looking at any result
log = {}

t0 = time.time()
g, stats = graph.load_snap("data/soc-Epinions1.txt.gz")
gc = graph.giant_component(g)
stats.update(giant_component_nodes=int(gc.sum()), mean_degree=float(g.deg[g.deg > 0].mean()), median_degree=float(np.median(g.deg[g.deg > 0])),
             max_degree=int(g.deg.max()), load_seconds=round(time.time() - t0, 1))
print("graph:", stats)

t0 = time.time()
X, extra = features.build(g)
stats.update(triangles=int(extra["support"].sum() // 3), max_core=int(extra["core"].max()), max_truss=int(extra["truss"].max()), feature_seconds=round(time.time() - t0, 1))
aug, plain = extra["forman_aug"], extra["forman_plain"]
stats.update(forman_plain_median=float(np.median(plain)), forman_aug_median=float(np.median(aug)), forman_aug_share_positive=float((aug > 0).mean()))
print("features:", X.shape, f"{stats['feature_seconds']} s")

# ---------- distances (BFS from 300 random giant-component nodes)
def distances(mask_nodes, indptr, nbr, k=300, seed=0):
    rng = np.random.default_rng(seed); src = rng.choice(np.where(mask_nodes)[0], k, replace=False); ds = []
    n = len(indptr) - 1
    for s in src:
        dist = np.full(n, -1); dist[s] = 0; frontier = [s]; d = 0
        while frontier:
            d += 1; nxt = []
            for v in frontier:
                for w in nbr[indptr[v]:indptr[v + 1]]:
                    if dist[w] < 0: dist[w] = d; nxt.append(w)
            frontier = nxt
        ds.append(dist[dist > 0])
    ds = np.concatenate(ds)
    return {"mean": float(ds.mean()), "p90": float(np.percentile(ds, 90)), "max_seen": int(ds.max())}
stats["distances_full_graph"] = distances(gc, g.indptr, g.nbr)
print("distances:", stats["distances_full_graph"])

# ---------- seeds: 1,000 giant-component nodes, 100 from each decile of degree rank
rng = np.random.default_rng(42)
cand = np.where(gc)[0]; order = cand[np.argsort(g.deg[cand], kind="stable")]
seeds = np.concatenate([rng.choice(part, N_SEEDS // 10, replace=False) for part in np.array_split(order, 10)])
beta_c = sir.epidemic_threshold(g); beta = BETA_MULT * beta_c
cache = os.path.join(OUT, "sir_cache.npz")
if os.path.exists(cache) and np.array_equal(np.load(cache)["seeds"], seeds):
    sizes = np.load(cache)["sizes"]; print("SIR: loaded cache")
else:
    t0 = time.time(); sizes = sir.outbreak_sizes(g, seeds, beta, RUNS, seed=1); stats["sir_seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(cache, seeds=seeds, sizes=sizes)
spread = sizes.mean(1); y = np.log(spread)
half_a, half_b = sizes[:, ::2].mean(1), sizes[:, 1::2].mean(1)
from scipy.stats import spearmanr, pearsonr
stats.update(beta_c=float(beta_c), beta=float(beta), runs_per_seed=RUNS, spread_median=float(np.median(spread)), spread_max=float(spread.max()),
             noise_ceiling_spearman_halves=float(spearmanr(half_a, half_b)[0]), noise_ceiling_r2_log_halves=float(pearsonr(np.log(half_a), np.log(half_b))[0] ** 2))
print("SIR:", {k: stats[k] for k in ("beta_c", "beta", "spread_median", "spread_max", "noise_ceiling_spearman_halves", "noise_ceiling_r2_log_halves")})
label = (spread >= np.quantile(spread, 1 - TOP)).astype(int)

# ---------- models, all scored out of fold
Xs = X[seeds]; idx = {n: i for i, n in enumerate(features.FEATURES)}
rows = []
for name in ["degree", "coreness", "trussness", "triangles", "collective_influence_2", "pagerank", "two_hop_reach"]:
    rows.append({"model": name + " alone", **unfitted(Xs[:, idx[name]], y, label)})
rows.append({"model": "-forman_mean alone", **unfitted(-Xs[:, idx["forman_mean"]], y, label)})
rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable") / (len(v) - 1)
dct = Xs[:, [idx["degree"], idx["coreness"], idx["trussness"]]]
rows.append({"model": "D-C-T rank average (no fitting)", **unfitted(rank(dct[:, 0]) + rank(dct[:, 1]) + rank(dct[:, 2]), y, label)})
rows.append({"model": "D-C-T ensemble (logistic / linear, 5-fold)", **oof(dct, y, label)})
rows.append({"model": "all 20 features (logistic / linear, 5-fold)", **oof(Xs, y, label)})
res = pd.DataFrame(rows); res.to_csv(os.path.join(T, "models.csv"), index=False)
print(res.round(4).to_string(index=False))

# ---------- robustness of the headline model to the infection rate (fewer runs, so noisier)
rob = []
for mult in (1.0, 2.0):
    s2 = sir.outbreak_sizes(g, seeds, mult * beta_c, 200, seed=7).mean(1)
    l2 = (s2 >= np.quantile(s2, 1 - TOP)).astype(int)
    rob.append({"beta_multiple": mult, "runs": 200, "spread_median": float(np.median(s2)), **oof(dct, np.log(s2), l2), "degree_auc": float(unfitted(Xs[:, 0], np.log(s2), l2)["roc_auc"])})
pd.DataFrame(rob).to_csv(os.path.join(T, "robustness_beta.csv"), index=False)
print(pd.DataFrame(rob).round(4).to_string(index=False))

# ---------- PCA: raw (saturated) vs signed-log + standardised
def shares(M):
    M = M - M.mean(0); s = np.linalg.svd(M, compute_uv=False) ** 2; return s / s.sum()
raw = shares(X[gc]); Z = transform(X[gc]); fixed = shares((Z - Z.mean(0)) / Z.std(0))
pca = {"raw_pc1": float(raw[0]), "raw_components_for_90pct": int(np.searchsorted(np.cumsum(raw), 0.9) + 1),
       "logstd_pc1": float(fixed[0]), "logstd_pc2": float(fixed[1]), "logstd_components_for_90pct": int(np.searchsorted(np.cumsum(fixed), 0.9) + 1),
       "logstd_shares": [round(float(v), 4) for v in fixed]}
stats["pca"] = pca; print("PCA:", pca)
Zs = transform(Xs); Zs = (Zs - Zs.mean(0)) / Zs.std(0)
U, S, Vt = np.linalg.svd(Zs, full_matrices=False)
for k in (1, 2, 3, 5, 10, 20):
    rows_k = oof(Zs @ Vt[:k].T, y, label)
    stats.setdefault("pca_components_model", []).append({"k": k, "roc_auc": round(rows_k["roc_auc"], 4), "r2": round(rows_k["r2"], 4)})
print("PCA components model:", stats["pca_components_model"])
load = pd.DataFrame(Vt[:3].T, index=features.FEATURES, columns=["PC1", "PC2", "PC3"]).round(3); load.to_csv(os.path.join(T, "pca_loadings.csv"))

# ---------- Forman-Ricci filtering: keep only nodes touching a strongly negatively curved edge
fmin = X[:, idx["forman_min"]]; deg = g.deg
top_nodes = seeds[label == 1]
filt = []
for q in (0.05, 0.10, 0.20, 0.30):
    thr = np.quantile(fmin[gc], q)
    keep_c = gc & (fmin <= thr)
    kdeg = np.sort(deg[gc])[::-1][keep_c.sum() - 1]; keep_d = gc & (deg >= kdeg)
    filt.append({"kept_share_of_graph": float(keep_c.sum() / gc.sum()), "curvature_threshold": float(thr),
                 "top_spreaders_retained_curvature": float(np.isin(top_nodes, np.where(keep_c)[0]).mean()),
                 "top_spreaders_retained_degree_same_size": float(np.isin(top_nodes, np.where(keep_d)[0]).mean()),
                 "seed_sample_retained": float(np.isin(seeds, np.where(keep_c)[0]).mean())})
pd.DataFrame(filt).to_csv(os.path.join(T, "forman_filter.csv"), index=False)
print(pd.DataFrame(filt).round(3).to_string(index=False))
# distances inside the filtered "periphery-removed" subgraph and in the backbone of positively curved edges
keep = gc & (fmin <= np.quantile(fmin[gc], 0.20))
sub_mask_edges = keep[g.eu] & keep[g.ev]
def sub_csr(mask_e):
    eu, ev = g.eu[mask_e], g.ev[mask_e]; sym = np.vstack([np.c_[eu, ev], np.c_[ev, eu]]); sym = sym[np.lexsort((sym[:, 1], sym[:, 0]))]
    ip = np.zeros(g.n + 1, np.int64); np.add.at(ip, sym[:, 0] + 1, 1); return np.cumsum(ip), sym[:, 1]
ip, nb = sub_csr(sub_mask_edges); dsub = np.diff(ip) > 0
stats["distances_curvature_filtered_20pct"] = distances(dsub, ip, nb)
ip2, nb2 = sub_csr(aug > 0); d2 = np.diff(ip2) > 0
stats["positive_curvature_backbone"] = {"edges": int((aug > 0).sum()), "nodes": int(d2.sum()), **distances(d2, ip2, nb2, k=min(300, int(d2.sum())))}
print("filtered distances:", stats["distances_curvature_filtered_20pct"], "| positive backbone:", stats["positive_curvature_backbone"])

pd.DataFrame(X, columns=features.FEATURES).iloc[seeds].assign(node=seeds, spread=spread, influencer=label).to_csv(os.path.join(T, "seed_table.csv"), index=False)
json.dump(stats, open(os.path.join(T, "summary.json"), "w"), indent=2)

# ---------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
d = deg[deg > 0]; vals, cnt = np.unique(d, return_counts=True); ax[0].loglog(vals, cnt / cnt.sum(), ".", ms=3); ax[0].set(xlabel="degree", ylabel="share of nodes", title="Degree distribution (log-log)")
ax[1].hist(aug, bins=200, log=True); ax[1].set(xlabel="augmented Forman-Ricci curvature", ylabel="edges", title="Edge curvature")
ax[2].scatter(deg[seeds], spread, c=label, s=6, cmap="coolwarm"); ax[2].set(xscale="log", yscale="log", xlabel="degree", ylabel="mean outbreak size", title=f"SIR spread, beta = {BETA_MULT} x threshold")
fig.tight_layout(); fig.savefig(os.path.join(F, "overview.png"), dpi=130)
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); ax[0].bar(range(1, 21), raw, color="grey"); ax[0].set(title="PCA on raw features", xlabel="component", ylabel="explained variance")
ax[1].bar(range(1, 21), fixed); ax[1].set(title="PCA after signed log + standardisation", xlabel="component"); fig.tight_layout(); fig.savefig(os.path.join(F, "pca.png"), dpi=130)
print("done")
