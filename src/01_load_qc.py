"""
01_load_qc.py — irAE scRNA-seq (GSE322576, 2026)
PBMC, CellRanger v3, 4 conditions: irAE / HC / RAC / ICI
Primary comparison: irAE vs HC
"""
import os, gc, re
import numpy as np
import pandas as pd
import scipy.io
import anndata as ad
import scanpy as sc

DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/irae_scrna_output"
RAW_DIR    = os.path.join(DRIVE_BASE, "raw")
PROCESSED  = os.path.join(DRIVE_BASE, "processed")
os.makedirs(PROCESSED, exist_ok=True)

CONDITION_MAP = {
    "GSM9555167": ("irAE","P1"),  "GSM9555168": ("HC","P2"),
    "GSM9555169": ("RAC","P3"),   "GSM9555170": ("RAC","P4"),
    "GSM9555171": ("HC","P5"),    "GSM9555172": ("RAC","P6"),
    "GSM9555173": ("RAC","P7"),   "GSM9555174": ("HC","P8"),
    "GSM9555175": ("HC","P9"),    "GSM9555176": ("HC","P10"),
    "GSM9555177": ("HC","P11"),   "GSM9555178": ("ICI","P12"),
    "GSM9555179": ("ICI","P13"),  "GSM9555180": ("irAE","P14"),
    "GSM9555181": ("irAE","P15"), "GSM9555182": ("RAC","P16"),
    "GSM9555183": ("irAE","P17"), "GSM9555184": ("RAC","P18"),
    "GSM9555185": ("irAE","P19"), "GSM9587734": ("irAE","P20"),
    "GSM9587735": ("HC","P21"),   "GSM9587736": ("RAC","P22"),
    "GSM9587737": ("RAC","P23"),  "GSM9587738": ("HC","P24"),
    "GSM9587739": ("RAC","P25"),  "GSM9587740": ("RAC","P26"),
    "GSM9587741": ("HC","P27"),   "GSM9587742": ("HC","P28"),
    "GSM9587743": ("HC","P29"),   "GSM9587744": ("HC","P30"),
    "GSM9587745": ("ICI","P31"),  "GSM9587746": ("ICI","P32"),
    "GSM9587747": ("irAE","P33"), "GSM9587748": ("irAE","P34"),
    "GSM9587749": ("RAC","P35"),  "GSM9587750": ("irAE","P36"),
    "GSM9587751": ("RAC","P37"),  "GSM9587752": ("irAE","P38"),
}

MIN_GENES_PREFILTER = 200
MIN_GENES  = 200
MAX_GENES  = 6000
MAX_MT_PCT = 20


def discover_samples(raw_dir):
    """CellRanger v3 flat layout: {GSM}_{tag}_{barcodes|features|matrix}.tsv/mtx.gz"""
    samples = {}
    for fname in os.listdir(raw_dir):
        m = re.match(r'^(GSM\d+)_.*_(barcodes|features|matrix)\.(tsv|mtx)\.gz$', fname)
        if not m: continue
        gsm, ftype = m.group(1), m.group(2)
        if gsm not in samples: samples[gsm] = {}
        samples[gsm][ftype] = os.path.join(raw_dir, fname)
    return samples


def load_sample_v3(gsm, paths, condition, patient):
    mat      = scipy.io.mmread(paths["matrix"]).T.tocsr()
    barcodes = pd.read_csv(paths["barcodes"], header=None, sep="\t")[0].tolist()
    features = pd.read_csv(paths["features"], header=None, sep="\t")
    keep     = np.diff(mat.indptr) >= MIN_GENES_PREFILTER
    mat      = mat[keep, :]
    barcodes = [bc for bc, k in zip(barcodes, keep) if k]
    adata = ad.AnnData(X=mat)
    adata.obs_names            = [f"{gsm}_{bc}" for bc in barcodes]
    adata.var_names            = features[1].tolist()
    adata.var["gene_ids"]      = features[0].tolist()
    adata.var["feature_types"] = features[2].tolist()
    adata.obs["condition"]     = condition
    adata.obs["patient"]       = patient
    adata.obs["sample"]        = gsm
    adata.var_names_make_unique()
    return adata


print("Discovering samples...")
all_samples = discover_samples(RAW_DIR)
print(f"  Found {len(all_samples)} GSM IDs")
unknown = [g for g in all_samples if g not in CONDITION_MAP]
if unknown: print(f"  WARNING — no condition map for: {unknown}")

adatas_all = []
for cond in ["irAE", "HC", "RAC", "ICI"]:
    print(f"\nLoading {cond} samples...")
    adatas = []
    for gsm in sorted(all_samples):
        if gsm not in CONDITION_MAP: continue
        c, patient = CONDITION_MAP[gsm]
        if c != cond: continue
        paths = all_samples[gsm]
        if not all(k in paths for k in ["barcodes","features","matrix"]):
            print(f"  {gsm} — SKIP (missing files)"); continue
        try:
            a = load_sample_v3(gsm, paths, cond, patient)
            print(f"  {gsm} (patient {patient}) — {a.n_obs:,} cells")
            adatas.append(a)
        except Exception as e:
            print(f"  {gsm} — ERROR: {e}")
    if not adatas:
        print(f"  WARNING: no {cond} samples loaded"); continue
    batch = sc.concat(adatas, join="outer", fill_value=0)
    del adatas; gc.collect()
    print(f"  {cond}: {batch.n_obs:,} cells")
    adatas_all.append(batch)

print("\nMerging all conditions...")
adata = sc.concat(adatas_all, join="outer", fill_value=0)
del adatas_all; gc.collect()
print(f"Combined: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(adata.obs["condition"].value_counts().to_string())

print("\nRunning QC...")
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
print(f"  Median genes/cell: {adata.obs['n_genes_by_counts'].median():.0f}")
print(f"  Median MT%:        {adata.obs['pct_counts_mt'].median():.1f}%")

n_before = adata.n_obs
adata = adata[adata.obs["n_genes_by_counts"] >= MIN_GENES].copy()
adata = adata[adata.obs["n_genes_by_counts"] <= MAX_GENES].copy()
adata = adata[adata.obs["pct_counts_mt"]     <= MAX_MT_PCT].copy()
print(f"  After QC: {adata.n_obs:,} cells ({n_before-adata.n_obs:,} removed)")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="sample", subset=True)
print(f"  HVGs: {adata.n_vars}")

out_path = os.path.join(PROCESSED, "01_loaded.h5ad")
adata.write_h5ad(out_path)
print(f"\nSaved: {out_path}")
print("Script 01 complete.")
