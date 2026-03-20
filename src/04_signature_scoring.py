"""
04_signature_scoring.py — irAE disease signature scoring across 4 conditions
"""
import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/irae_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

IRAE_SIGNATURES = {
    "t_cell_exhaustion_activation": [
        "PDCD1","LAG3","TIGIT","HAVCR2","TOX","GZMK","GZMB","IFNG",
        "PRF1","NKG7","ENTPD1","CTLA4","CD38","CXCR3"],
    "th1_th17_inflammation": [
        "IFNG","IL17A","CXCL10","CXCL9","STAT1","IRF1","TBX21",
        "RORC","IL6","IL18","IL23A","IL12A","CCL5","CXCL13"],
    "checkpoint_dysregulation": [
        "PDCD1","CTLA4","CD274","PDCD1LG2","CD80","CD86",
        "ICOS","ICOSLG","LAG3","TIGIT","HAVCR2","VSIR"],
    "cytokine_storm": [
        "IL1B","TNF","CXCL8","IL6","CCL2","CCL5","CXCL1",
        "IL18","IL23A","IL12A","CXCL9","CXCL10","IL1A"],
    "isg_interferon_response": [
        "ISG15","MX1","OAS1","IFIT1","IFIT3","IFI44L",
        "STAT1","IRF1","CXCL10","CXCL9","IFI6","RSAD2","HERC5"],
}

in_path  = os.path.join(PROCESSED, "03_annotated.h5ad")
out_path = os.path.join(PROCESSED, "04_scored.h5ad")

if not os.path.exists(in_path):
    print("ERROR: 03_annotated.h5ad not found. Run 03_annotate_clusters.py first.")
    sys.exit(1)

print("Loading 03_annotated.h5ad...")
adata = sc.read_h5ad(in_path)
print(f"  {adata.n_obs:,} cells x {adata.n_vars} genes")
if "norm_log" not in adata.layers:
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)

clusters = sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
cluster_scores = {sig: [] for sig in IRAE_SIGNATURES}
print("Scoring irAE signatures:")
for sig_name, genes in IRAE_SIGNATURES.items():
    present = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    msg = f"  {sig_name}: {len(present)}/{len(genes)} genes"
    if missing: msg += f" (missing: {missing[:3]}{'...' if len(missing)>3 else ''})"
    print(msg)
    if not present:
        adata.obs["score_"+sig_name] = 0.0
        cluster_scores[sig_name].extend([0.0]*len(clusters)); continue
    e = adata[:, present].X
    if hasattr(e, "toarray"): e = e.toarray()
    cell_scores = np.array(e.mean(axis=1)).flatten()
    adata.obs["score_"+sig_name] = cell_scores
    for cl in clusters:
        mask = adata.obs["leiden"] == cl
        cluster_scores[sig_name].append(float(cell_scores[mask].mean()))

sig_df = pd.DataFrame(cluster_scores, index=clusters)
sig_df.index.name = "cluster"
sig_df["irae_primary_score"] = (sig_df.get("t_cell_exhaustion_activation", 0) +
                                 sig_df.get("th1_th17_inflammation", 0))
sig_df["irae_composite_score"] = sig_df[[c for c in IRAE_SIGNATURES
                                          if c in sig_df.columns]].sum(axis=1)

top_clusters = sig_df.nlargest(3, "irae_primary_score")
print("\nTop 3 pro-irAE clusters:")
for cl, row in top_clusters.iterrows():
    ct = adata.obs.loc[adata.obs["leiden"]==str(cl), "cell_type"].values
    ct = ct[0] if len(ct) > 0 else "unknown"
    print(f"  Cluster {cl} ({ct}): exhaustion/act={row.get('t_cell_exhaustion_activation',0):.4f}, "
          f"th1/th17={row.get('th1_th17_inflammation',0):.4f}")

# Scores by condition — all 4 groups
cond_rows = []
for cond in ["irAE", "HC", "RAC", "ICI"]:
    mask = adata.obs["condition"] == cond
    if not mask.any(): continue
    row = {"condition": cond, "n_cells": int(mask.sum())}
    for sig in IRAE_SIGNATURES:
        col = "score_"+sig
        row[sig] = round(float(adata.obs.loc[mask, col].mean()), 4) \
                   if col in adata.obs.columns else 0.0
    cond_rows.append(row)
cond_df = pd.DataFrame(cond_rows)
print("\nScores by condition:")
print(cond_df[["condition","n_cells","t_cell_exhaustion_activation",
               "th1_th17_inflammation","isg_interferon_response"]].round(4).to_string(index=False))

adata.uns["pro_irae_clusters"] = top_clusters.index.tolist()
adata.write_h5ad(out_path)
sig_df.to_csv(os.path.join(PROCESSED, "signature_scores.csv"))
cond_df.to_csv(os.path.join(PROCESSED, "signature_scores_by_condition.csv"), index=False)
print(f"\nSaved: {out_path}")
print("Script 04 complete.")
