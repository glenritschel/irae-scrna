"""
03_annotate_clusters.py — Marker-based PBMC cell type annotation
"""
import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/irae_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

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
CONFIDENCE_THRESHOLD = 1.2

in_path  = os.path.join(PROCESSED, "02_scvi.h5ad")
out_path = os.path.join(PROCESSED, "03_annotated.h5ad")

if not os.path.exists(in_path):
    print("ERROR: 02_scvi.h5ad not found. Run 02_scvi_embed.py first.")
    sys.exit(1)

print("Loading 02_scvi.h5ad...")
adata = sc.read_h5ad(in_path)
print(f"  {adata.n_obs:,} cells, {adata.obs['leiden'].nunique()} clusters")
if "norm_log" not in adata.layers:
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)

clusters = sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
scores = {ct: [] for ct in CELL_TYPE_MARKERS}
for cl in clusters:
    mask = adata.obs["leiden"] == cl
    for ct, markers in CELL_TYPE_MARKERS.items():
        present = [m for m in markers if m in adata.var_names]
        if not present: scores[ct].append(0.0); continue
        e = adata[mask, present].X
        if hasattr(e, "toarray"): e = e.toarray()
        scores[ct].append(float(e.mean()))

score_df = pd.DataFrame(scores, index=clusters)
score_df.index.name = "cluster"

ann_rows = []
for cluster in score_df.index:
    row = score_df.loc[cluster]
    best = row.idxmax(); best_score = row.max()
    runner_up = row.sort_values(ascending=False).iloc[1] if len(row) > 1 else 0.0
    if best_score == 0.0:
        label, conf = "unknown", "low"
    elif runner_up == 0.0 or best_score >= CONFIDENCE_THRESHOLD * runner_up:
        label, conf = best, "high"
    else:
        label, conf = best + "_mixed", "low"
    ann_rows.append({"cluster": cluster, "annotation": label, "confidence": conf,
                     "best_score": round(best_score, 4),
                     "runner_up_score": round(runner_up, 4)})

ann_df = pd.DataFrame(ann_rows)
counts  = adata.obs["leiden"].value_counts().rename("n_cells")
summary = ann_df.set_index("cluster").join(counts).sort_values("n_cells", ascending=False)
print("\nCluster annotations:")
print("-" * 75)
for cl, row in summary.iterrows():
    print(f"  Cluster {cl:>3} | {row['annotation']:<25} | {row['confidence']:<4} | {int(row['n_cells']):>6} cells")

adata.obs["cell_type"] = adata.obs["leiden"].map(
    dict(zip(ann_df["cluster"].astype(str), ann_df["annotation"])))
adata.obs["annotation_confidence"] = adata.obs["leiden"].map(
    dict(zip(ann_df["cluster"].astype(str), ann_df["confidence"])))

adata.write_h5ad(out_path)
ann_df.to_csv(os.path.join(PROCESSED, "cluster_annotations.csv"), index=False)
score_df.to_csv(os.path.join(PROCESSED, "cluster_marker_scores.csv"))
print(f"\nSaved: {out_path}")
print("Script 03 complete.")
