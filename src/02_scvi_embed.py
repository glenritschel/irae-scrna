"""
02_scvi_embed.py — scVI embedding + Leiden clustering
"""
import os, sys, random
import numpy as np
import pandas as pd
import scanpy as sc

DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/irae_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")
os.makedirs(PROCESSED, exist_ok=True)

SCVI_PARAMS        = {"n_latent": 30, "n_layers": 2, "n_hidden": 128}
SCVI_TRAIN_PARAMS  = {"max_epochs": 200, "early_stopping": True, "early_stopping_patience": 20}
N_NEIGHBORS        = 15
RANDOM_SEED        = 0
LEIDEN_RESOLUTIONS = [0.5, 0.8, 1.2]

CELL_TYPE_MARKERS = {
    "cd4_t_cell":   ["CD3D","CD3E","CD4","IL7R","TRAC"],
    "cd8_t_cell":   ["CD3D","CD3E","CD8A","CD8B","GZMK","GZMB"],
    "treg":         ["FOXP3","CTLA4","IL7R","CD4","IKZF2"],
    "exhausted_t":  ["PDCD1","LAG3","TIGIT","HAVCR2","TOX","ENTPD1"],
    "nk_cell":      ["GNLY","NKG7","NCAM1","KLRD1","KLRB1","GZMB"],
    "b_cell":       ["CD79A","MS4A1","CD19","IGHM","IGHD","CD27"],
    "plasma_cell":  ["MZB1","JCHAIN","IGHG1","IGHA1","SDC1"],
    "monocyte":     ["CD14","LYZ","S100A8","S100A9","CCL2","FCGR3A"],
    "dc":           ["HLA-DRA","CLEC9A","FCER1A","CLEC10A","IRF8"],
}

in_path  = os.path.join(PROCESSED, "01_loaded.h5ad")
ckpt     = os.path.join(PROCESSED, "02_scvi_ckpt.h5ad")
out_path = os.path.join(PROCESSED, "02_scvi.h5ad")

if not os.path.exists(in_path):
    print("ERROR: 01_loaded.h5ad not found. Run 01_load_qc.py first.")
    sys.exit(1)

import scvi, torch
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
scvi.settings.seed = RANDOM_SEED

if os.path.exists(ckpt):
    print("Loading from post-training checkpoint...")
    adata = sc.read_h5ad(ckpt)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars} HVGs")
    print(f"  X_scVI present: {'X_scVI' in adata.obsm}")
else:
    print("Loading 01_loaded.h5ad...")
    adata = sc.read_h5ad(in_path)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars} HVGs")
    scvi.model.SCVI.setup_anndata(adata, batch_key="sample")
    model = scvi.model.SCVI(adata, **SCVI_PARAMS)
    print(f"Training scVI on {adata.n_obs:,} cells...")
    model.train(**SCVI_TRAIN_PARAMS, accelerator="auto")
    try:
        key = "train_loss_epoch" if "train_loss_epoch" in model.history else "train_loss"
        print(f"Training complete. Final loss: {float(np.array(model.history[key].values[-1]).flat[0]):.2f}")
    except Exception:
        print("Training complete.")
    adata.obsm["X_scVI"] = model.get_latent_representation()
    print(f"Latent shape: {adata.obsm['X_scVI'].shape}")
    adata.write_h5ad(ckpt)
    print(f"Checkpoint saved: {ckpt}")

print("\nBuilding neighbour graph and UMAP...")
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=N_NEIGHBORS)
sc.tl.umap(adata)

print("\nRunning Leiden at multiple resolutions...")
res_results = {}
for res in LEIDEN_RESOLUTIONS:
    key = f"leiden_{res}"
    sc.tl.leiden(adata, resolution=res, key_added=key,
                 random_state=RANDOM_SEED, flavor="igraph",
                 n_iterations=2, directed=False)
    n = adata.obs[key].nunique()
    print(f"  Resolution {res}: {n} clusters")
    res_results[res] = {"key": key, "n_clusters": n}

rows = []
for res, info in res_results.items():
    best_clusters = {}
    for ct, markers in CELL_TYPE_MARKERS.items():
        present = [m for m in markers if m in adata.var_names]
        if not present: continue
        cluster_means = {}
        for cl in adata.obs[info["key"]].unique():
            mask = adata.obs[info["key"]] == cl
            e = adata[mask, present].X
            if hasattr(e, "toarray"): e = e.toarray()
            cluster_means[cl] = float(e.mean())
        best_clusters[ct] = max(cluster_means, key=cluster_means.get)
    n_distinct = len(set(best_clusters.values()))
    n_resolved = len(best_clusters)
    rows.append({"resolution": res, "n_clusters": info["n_clusters"],
                 "n_celltypes_resolved": n_resolved,
                 "n_distinct_best_clusters": n_distinct,
                 "separation_score": n_distinct / max(n_resolved, 1)})

res_df = pd.DataFrame(rows).sort_values("separation_score", ascending=False)
print("\nResolution comparison:")
print(res_df.to_string(index=False))
recommended = float(res_df.iloc[0]["resolution"])
print(f"\nRecommended: {recommended} ({int(res_df.iloc[0]['n_clusters'])} clusters)")

adata.obs["leiden"] = adata.obs[f"leiden_{recommended}"].copy()
adata.uns["recommended_leiden_resolution"] = recommended
adata.uns["pro_irae_clusters"] = []

adata.write_h5ad(out_path)
res_df.to_csv(os.path.join(PROCESSED, "resolution_metrics.csv"), index=False)
adata.obs.groupby(["leiden","condition"]).size().unstack(fill_value=0).to_csv(
    os.path.join(PROCESSED, "cluster_condition_distribution.csv"))
print(f"\nSaved: {out_path}")
print(f"Clusters: {adata.obs['leiden'].nunique()}")
print("Script 02 complete.")
