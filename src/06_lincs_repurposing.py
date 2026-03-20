"""
06_lincs_repurposing.py — LINCS L1000 transcriptomic reversal scoring
Primary queries: irAE vs HC. Secondary: irAE vs RAC, irAE vs ICI.
"""
import os, sys, re, time
import numpy as np
import pandas as pd
import gseapy as gp

DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/irae_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

N_TOP_GENES_LINCS = 150
TOP_PER_QUERY     = 15
ENRICHR_DELAY     = 1.0
ENRICHR_LIBRARIES = [
    "LINCS_L1000_Chem_Pert_up",
    "LINCS_L1000_Chem_Pert_down",
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "KEGG_2021_Human",
]


def clean_compound_name(term):
    m = re.match(r"^LJP\d+\s+\S+\s+\S+?-(.+)-[\d.]+$", term.strip())
    if m: return m.group(1).strip()
    return term.split("_")[0].strip() if term else term.strip()


def run_enrichr(query_id, up_genes, down_genes):
    results = []
    for direction, genes, reversal_lib in [
        ("up",   up_genes,   "LINCS_L1000_Chem_Pert_down"),
        ("down", down_genes, "LINCS_L1000_Chem_Pert_up"),
    ]:
        if not genes: continue
        for lib in ENRICHR_LIBRARIES:
            try:
                enr = gp.enrichr(gene_list=genes, gene_sets=lib, outdir=None, verbose=False)
                df = enr.results.copy()
                if df.empty: continue
                df["query_id"]        = query_id
                df["query_direction"] = direction
                df["library"]         = lib
                if lib in ("LINCS_L1000_Chem_Pert_up", "LINCS_L1000_Chem_Pert_down"):
                    adj_p = df["Adjusted P-value"].clip(lower=1e-300)
                    sign  = 1.0 if lib == reversal_lib else -1.0
                    df["reversal_score"] = sign * (-np.log10(adj_p))
                    df["compound"] = df["Term"].apply(clean_compound_name)
                else:
                    df["reversal_score"] = 0.0
                    df["compound"] = df["Term"]
                results.append(df)
                time.sleep(ENRICHR_DELAY)
            except Exception as e:
                print(f"    WARNING: [{query_id} {lib}]: {e}")
                time.sleep(ENRICHR_DELAY * 2)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# Load DE files
for fname in ["de_irAE_vs_HC.csv", "de_top_genes.csv"]:
    if not os.path.exists(os.path.join(PROCESSED, fname)):
        print(f"ERROR: {fname} not found. Run 05_differential_expression.py first.")
        sys.exit(1)

print("Loading DE gene lists...")
irae_vs_hc   = pd.read_csv(os.path.join(PROCESSED, "de_irAE_vs_HC.csv"))
irae_vs_rac  = pd.read_csv(os.path.join(PROCESSED, "de_irAE_vs_RAC.csv")) \
               if os.path.exists(os.path.join(PROCESSED, "de_irAE_vs_RAC.csv")) else None
irae_vs_ici  = pd.read_csv(os.path.join(PROCESSED, "de_irAE_vs_ICI.csv")) \
               if os.path.exists(os.path.join(PROCESSED, "de_irAE_vs_ICI.csv")) else None
top_genes_df = pd.read_csv(os.path.join(PROCESSED, "de_top_genes.csv"))
print(f"  {len(irae_vs_hc):,} genes | {top_genes_df['cluster'].nunique()} clusters")

all_results = []

# Primary query: irAE vs HC
print("Primary query: irAE vs HC...")
res = run_enrichr("irAE_vs_HC",
                  irae_vs_hc[irae_vs_hc["score"]>0].head(N_TOP_GENES_LINCS)["gene"].tolist(),
                  irae_vs_hc[irae_vs_hc["score"]<0].tail(N_TOP_GENES_LINCS)["gene"].tolist())
if not res.empty:
    all_results.append(res)
    print(f"  {(res['reversal_score']>0).sum()} reversal hits")

# Secondary: irAE vs RAC
if irae_vs_rac is not None:
    print("Secondary query: irAE vs RAC...")
    res = run_enrichr("irAE_vs_RAC",
                      irae_vs_rac[irae_vs_rac["score"]>0].head(N_TOP_GENES_LINCS)["gene"].tolist(),
                      irae_vs_rac[irae_vs_rac["score"]<0].tail(N_TOP_GENES_LINCS)["gene"].tolist())
    if not res.empty:
        all_results.append(res)
        print(f"  {(res['reversal_score']>0).sum()} reversal hits")

# Secondary: irAE vs ICI
if irae_vs_ici is not None:
    print("Secondary query: irAE vs ICI...")
    res = run_enrichr("irAE_vs_ICI",
                      irae_vs_ici[irae_vs_ici["score"]>0].head(N_TOP_GENES_LINCS)["gene"].tolist(),
                      irae_vs_ici[irae_vs_ici["score"]<0].tail(N_TOP_GENES_LINCS)["gene"].tolist())
    if not res.empty:
        all_results.append(res)
        print(f"  {(res['reversal_score']>0).sum()} reversal hits")

# T cell subset query
tcell_path = os.path.join(PROCESSED, "de_tcell_irAE_vs_HC.csv")
if os.path.exists(tcell_path):
    tcell_de = pd.read_csv(tcell_path)
    res = run_enrichr("Tcell_irAE_vs_HC",
                      tcell_de[tcell_de["score"]>0].head(N_TOP_GENES_LINCS)["gene"].tolist(),
                      tcell_de[tcell_de["score"]<0].tail(N_TOP_GENES_LINCS)["gene"].tolist())
    if not res.empty:
        all_results.append(res)
        print(f"  T cell: {(res['reversal_score']>0).sum()} hits")

# Pro-irAE cluster query
proirae_path = os.path.join(PROCESSED, "de_proirae_vs_rest.csv")
if os.path.exists(proirae_path):
    proirae_de = pd.read_csv(proirae_path)
    res = run_enrichr("ProirAE_clusters",
                      proirae_de[proirae_de["score"]>0].head(N_TOP_GENES_LINCS)["gene"].tolist(),
                      proirae_de[proirae_de["score"]<0].tail(N_TOP_GENES_LINCS)["gene"].tolist())
    if not res.empty:
        all_results.append(res)
        print(f"  Pro-irAE clusters: {(res['reversal_score']>0).sum()} hits")

# Cluster queries
n_cl = top_genes_df["cluster"].nunique()
for i, cl in enumerate(top_genes_df["cluster"].unique()):
    print(f"  Cluster {cl} ({i+1}/{n_cl})...", end=" ", flush=True)
    up   = top_genes_df.loc[(top_genes_df["cluster"]==cl) &
                             (top_genes_df["direction"]=="up"), "gene"].tolist()
    down = top_genes_df.loc[(top_genes_df["cluster"]==cl) &
                             (top_genes_df["direction"]=="down"), "gene"].tolist()
    res  = run_enrichr(f"cluster_{cl}", up, down)
    if not res.empty:
        all_results.append(res)
        print(f"{(res['reversal_score']>0).sum()} hits")
    else:
        print("no results")

if not all_results:
    print("ERROR: No Enrichr results returned.")
    sys.exit(1)

raw_results = pd.concat(all_results, ignore_index=True)
raw_results.to_csv(os.path.join(PROCESSED, "lincs_results_raw.csv"), index=False)
print(f"\nRaw results: {len(raw_results):,} rows")

lincs_df = raw_results[
    raw_results["library"].str.startswith("LINCS_L1000") &
    (raw_results["reversal_score"] > 0)].copy()
top = lincs_df.sort_values("reversal_score", ascending=False).groupby("query_id").head(TOP_PER_QUERY)
candidates = top.groupby("compound").agg(
    max_reversal_score=("reversal_score","max"),
    n_queries=("query_id","nunique"),
    queries=("query_id", lambda x: ",".join(sorted(set(x.astype(str))))),
).reset_index().sort_values("max_reversal_score", ascending=False)

print(f"Unique compounds: {len(candidates)}")
print(candidates[["compound","max_reversal_score","n_queries"]].head(15).round(2).to_string(index=False))
candidates.to_csv(os.path.join(PROCESSED, "lincs_candidates.csv"), index=False)
print("Script 06 complete.")
