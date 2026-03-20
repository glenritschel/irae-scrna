"""
05_differential_expression.py — Three-way DE: irAE vs HC (primary),
irAE vs RAC (novelty filter), irAE vs ICI (artifact filter)
"""
import os, sys
import pandas as pd
import scanpy as sc

DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/irae_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

N_TOP_GENES_DE = 150

in_path  = os.path.join(PROCESSED, "04_scored.h5ad")
out_path = os.path.join(PROCESSED, "05_de.h5ad")

if not os.path.exists(in_path):
    print("ERROR: 04_scored.h5ad not found. Run 04_signature_scoring.py first.")
    sys.exit(1)

print("Loading 04_scored.h5ad...")
adata = sc.read_h5ad(in_path)
print(f"  {adata.n_obs:,} cells, {adata.obs['leiden'].nunique()} clusters")
for cond in ["irAE","HC","RAC","ICI"]:
    n = (adata.obs["condition"]==cond).sum()
    print(f"  {cond}: {n:,}")
if "norm_log" not in adata.layers:
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)

# ── Primary: irAE vs HC ────────────────────────────────────────────────────
print("\nirAE vs HC DE (primary)...")
sc.tl.rank_genes_groups(adata, groupby="condition", groups=["irAE"],
                        reference="HC", method="wilcoxon",
                        use_raw=False, key_added="rank_genes_irae_vs_hc", pts=True)
r = adata.uns["rank_genes_irae_vs_hc"]
irae_vs_hc = pd.DataFrame({
    "gene": r["names"]["irAE"], "score": r["scores"]["irAE"],
    "pval_adj": r["pvals_adj"]["irAE"], "log2fc": r["logfoldchanges"]["irAE"],
}).sort_values("score", ascending=False)
irae_vs_hc.to_csv(os.path.join(PROCESSED, "de_irAE_vs_HC.csv"), index=False)
print(f"  Top up:   {irae_vs_hc.head(5)['gene'].tolist()}")
print(f"  Top down: {irae_vs_hc.tail(5)['gene'].tolist()}")

# ── Secondary: irAE vs RAC ─────────────────────────────────────────────────
print("\nirAE vs RAC DE (novelty filter)...")
sc.tl.rank_genes_groups(adata, groupby="condition", groups=["irAE"],
                        reference="RAC", method="wilcoxon",
                        use_raw=False, key_added="rank_genes_irae_vs_rac", pts=True)
r2 = adata.uns["rank_genes_irae_vs_rac"]
irae_vs_rac = pd.DataFrame({
    "gene": r2["names"]["irAE"], "score": r2["scores"]["irAE"],
    "pval_adj": r2["pvals_adj"]["irAE"], "log2fc": r2["logfoldchanges"]["irAE"],
}).sort_values("score", ascending=False)
irae_vs_rac.to_csv(os.path.join(PROCESSED, "de_irAE_vs_RAC.csv"), index=False)
print(f"  Top up:   {irae_vs_rac.head(5)['gene'].tolist()}")

# ── Secondary: irAE vs ICI ─────────────────────────────────────────────────
print("\nirAE vs ICI DE (artifact filter)...")
sc.tl.rank_genes_groups(adata, groupby="condition", groups=["irAE"],
                        reference="ICI", method="wilcoxon",
                        use_raw=False, key_added="rank_genes_irae_vs_ici", pts=True)
r3 = adata.uns["rank_genes_irae_vs_ici"]
irae_vs_ici = pd.DataFrame({
    "gene": r3["names"]["irAE"], "score": r3["scores"]["irAE"],
    "pval_adj": r3["pvals_adj"]["irAE"], "log2fc": r3["logfoldchanges"]["irAE"],
}).sort_values("score", ascending=False)
irae_vs_ici.to_csv(os.path.join(PROCESSED, "de_irAE_vs_ICI.csv"), index=False)
print(f"  Top up:   {irae_vs_ici.head(5)['gene'].tolist()}")

# ── Cluster-vs-rest ────────────────────────────────────────────────────────
print("\nCluster-vs-rest DE...")
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                        use_raw=False, key_added="rank_genes_groups", pts=True)
r4 = adata.uns["rank_genes_groups"]
rows_de = []
for cl in r4["names"].dtype.names:
    df_cl = pd.DataFrame({
        "cluster": cl, "gene": r4["names"][cl],
        "score": r4["scores"][cl], "pval_adj": r4["pvals_adj"][cl]})
    rows_de.extend([df_cl.nlargest(N_TOP_GENES_DE, "score").assign(direction="up"),
                    df_cl.nsmallest(N_TOP_GENES_DE, "score").assign(direction="down")])
top_genes_df = pd.concat(rows_de, ignore_index=True)
top_genes_df.to_csv(os.path.join(PROCESSED, "de_top_genes.csv"), index=False)
print(f"  {len(top_genes_df):,} gene-cluster pairs")

# ── Pro-irAE cluster DE ────────────────────────────────────────────────────
pro_clusters = [str(c) for c in list(adata.uns.get("pro_irae_clusters", []))]
if pro_clusters:
    adata.obs["pro_irae_group"] = adata.obs["leiden"].apply(
        lambda x: "pro_irae" if str(x) in pro_clusters else "other")
    sc.tl.rank_genes_groups(adata, groupby="pro_irae_group", groups=["pro_irae"],
                            reference="other", method="wilcoxon",
                            use_raw=False, key_added="rank_genes_proirae")
    pr = adata.uns["rank_genes_proirae"]
    pd.DataFrame({"gene": pr["names"]["pro_irae"], "score": pr["scores"]["pro_irae"],
                  "pval_adj": pr["pvals_adj"]["pro_irae"]}
                ).sort_values("score", ascending=False
                ).to_csv(os.path.join(PROCESSED, "de_proirae_vs_rest.csv"), index=False)
    print("  Pro-irAE cluster DE saved.")

# ── T cell subset DE ───────────────────────────────────────────────────────
if "cell_type" in adata.obs.columns:
    tcell_mask = adata.obs["cell_type"].str.contains("t_cell|exhausted|treg", na=False)
    adata_tcell = adata[tcell_mask].copy()
    print(f"\nT cell subset: {adata_tcell.n_obs:,} cells")
    if adata_tcell.n_obs > 100 and (adata_tcell.obs["condition"]=="irAE").sum() > 50:
        sc.tl.rank_genes_groups(adata_tcell, groupby="condition", groups=["irAE"],
                                reference="HC", method="wilcoxon",
                                use_raw=False, key_added="rank_tcell_irae")
        tr = adata_tcell.uns["rank_tcell_irae"]
        pd.DataFrame({"gene": tr["names"]["irAE"], "score": tr["scores"]["irAE"],
                      "pval_adj": tr["pvals_adj"]["irAE"]}
                    ).sort_values("score", ascending=False
                    ).to_csv(os.path.join(PROCESSED, "de_tcell_irAE_vs_HC.csv"), index=False)
        print("  T cell DE saved.")
    else:
        print("  Insufficient T cells — skipping focused DE.")

adata.write_h5ad(out_path)
print(f"\nSaved: {out_path}")
print("Script 05 complete.")
