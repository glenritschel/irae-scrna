"""
07_novelty_prioritization.py — PubMed novelty + priority scoring
irAE-specific novelty queries: no prior publications in irAE context = NOVEL_ALL
"""
import os, sys, time
import pandas as pd
import requests

DRIVE_BASE = "/content/drive/MyDrive/Ritschel_Research/irae_scrna_output"
PROCESSED  = os.path.join(DRIVE_BASE, "processed")

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_DELAY   = 0.4
NCBI_EMAIL   = "glen.ritschel@ritschelresearch.com"
NOVELTY_WEIGHTS = {"NOVEL_ALL": 3.0, "NOVEL_IRAE": 1.5, "KNOWN": 1.0}
PATENT_WATCH_MIN_REVERSAL = 20.0
PATENT_WATCH_MIN_QUERIES  = 2

MOA_REFERENCE = {
    "wz-3105":"SRC/ABL inhibitor", "wz-4-145":"CDK8 inhibitor",
    "pf-431396":"FAK/PYK2 inhibitor", "ql-xii-47":"MELK/FLT3 inhibitor",
    "ql-x-138":"BTK/MNK inhibitor", "as-601245":"JNK inhibitor",
    "cgp-60474":"CDK1/2 inhibitor", "azd-7762":"CHK1/2 inhibitor",
    "at-7519":"CDK1/2/4/6/9 inhibitor", "alvocidib":"CDK1/2/4/6/9 inhibitor",
    "bi-2536":"PLK1 inhibitor", "xmd-1150":"ERK5 inhibitor",
    "canertinib":"Pan-EGFR inhibitor", "pelitinib":"Pan-EGFR inhibitor",
    "tofacitinib":"JAK1/3 inhibitor", "baricitinib":"JAK1/2 inhibitor",
    "upadacitinib":"JAK1 inhibitor", "ruxolitinib":"JAK1/2 inhibitor",
    "pd-0325901":"MEK1/2 inhibitor", "selumetinib":"MEK1/2 inhibitor",
    "dasatinib":"BCR-ABL/SRC inhibitor", "nilotinib":"BCR-ABL inhibitor",
    "ipilimumab":"CTLA4 inhibitor", "nivolumab":"PD-1 inhibitor",
    "pembrolizumab":"PD-1 inhibitor", "atezolizumab":"PD-L1 inhibitor",
    "fostamatinib":"SYK inhibitor", "ibrutinib":"BTK inhibitor",
    "abatacept":"CTLA4-Ig fusion", "tocilizumab":"IL-6R inhibitor",
    "infliximab":"TNF inhibitor", "adalimumab":"TNF inhibitor",
    "ixekizumab":"IL-17A inhibitor", "secukinumab":"IL-17A inhibitor",
    "guselkumab":"IL-23 inhibitor", "risankizumab":"IL-23 inhibitor",
    "rapamycin":"mTORC1 inhibitor", "methotrexate":"Antifolate",
    "celastrol":"NF-kB/HSP90 inhibitor", "radicicol":"HSP90 inhibitor",
}


def pubmed_hit_count(query, retries=3):
    params = {"db":"pubmed","term":query,"rettype":"count",
              "retmode":"json","email":NCBI_EMAIL}
    for attempt in range(retries):
        try:
            resp = requests.get(NCBI_ESEARCH, params=params, timeout=10)
            resp.raise_for_status()
            count = int(resp.json()["esearchresult"]["count"])
            time.sleep(NCBI_DELAY)
            return count
        except Exception:
            if attempt < retries - 1: time.sleep(NCBI_DELAY * 3)
            else: return -1


def assess_novelty(compound_name):
    q = f'"{compound_name}"'
    hits_irae    = pubmed_hit_count(q + ' AND ("immune-related adverse event" OR "irAE" OR "checkpoint inhibitor arthritis")')
    hits_immune  = pubmed_hit_count(q + ' AND ("PD-1" OR "CTLA4" OR "checkpoint inhibitor" OR "immunotherapy")')
    hits_arthrit = pubmed_hit_count(q + ' AND ("arthritis" OR "inflammatory arthritis" OR "synovitis")')
    if hits_irae == 0 and hits_immune == 0 and hits_arthrit == 0:
        tier = "NOVEL_ALL"
    elif hits_irae == 0 and hits_immune == 0:
        tier = "NOVEL_IRAE"
    else:
        tier = "KNOWN"
    return {"compound": compound_name,
            "hits_irae": hits_irae,
            "hits_immune_checkpoint": hits_immune,
            "hits_arthritis": hits_arthrit,
            "novelty_tier": tier}


cand_path = os.path.join(PROCESSED, "lincs_candidates.csv")
if not os.path.exists(cand_path):
    print("ERROR: lincs_candidates.csv not found. Run 06_lincs_repurposing.py first.")
    sys.exit(1)

print("Loading LINCS candidates...")
candidates = pd.read_csv(cand_path)
print(f"  {len(candidates)} candidates")

novelty_rows = []
for i, row in candidates.iterrows():
    compound = row["compound"]
    print(f"  [{i+1}/{len(candidates)}] {compound}...", end=" ", flush=True)
    nov = assess_novelty(compound)
    novelty_rows.append(nov)
    print(f"{nov['novelty_tier']} "
          f"(irAE:{nov['hits_irae']}, ICI:{nov['hits_immune_checkpoint']}, "
          f"Arth:{nov['hits_arthritis']})")

novelty_df = pd.DataFrame(novelty_rows)
novelty_df.to_csv(os.path.join(PROCESSED, "novelty_raw.csv"), index=False)

merged = candidates.merge(novelty_df, on="compound", how="left")
merged["novelty_tier"]   = merged["novelty_tier"].fillna("KNOWN")
merged["moa"]            = merged["compound"].apply(
    lambda x: MOA_REFERENCE.get(x.lower().strip(), "unknown"))
merged["priority_score"] = merged.apply(
    lambda r: round(r["max_reversal_score"] *
                    NOVELTY_WEIGHTS.get(r["novelty_tier"], 1.0) *
                    r["n_queries"], 1), axis=1)
merged = merged.sort_values("priority_score", ascending=False)

print("\nNovelty breakdown:")
print(merged["novelty_tier"].value_counts().to_string())

display_cols = ["compound","moa","novelty_tier","max_reversal_score","n_queries","priority_score"]
print("\nTop 20 priority candidates:")
print(merged[display_cols].head(20).round(2).to_string(index=False))

patent_watch = merged[
    (merged["novelty_tier"] == "NOVEL_ALL") &
    (merged["max_reversal_score"] >= PATENT_WATCH_MIN_REVERSAL) &
    (merged["n_queries"] >= PATENT_WATCH_MIN_QUERIES)
].copy()
print(f"\nPatent watch: {len(patent_watch)} NOVEL_ALL compounds")
if not patent_watch.empty:
    print(patent_watch[display_cols].to_string(index=False))

merged.to_csv(os.path.join(PROCESSED, "priority_candidates.csv"), index=False)
patent_watch.to_csv(os.path.join(PROCESSED, "patent_watch.csv"), index=False)
print(f"\nSaved: priority_candidates.csv ({len(merged)} compounds)")
print(f"Saved: patent_watch.csv ({len(patent_watch)} compounds)")
print("\nScript 07 complete.")
print("=" * 60)
print("PIPELINE COMPLETE — review priority_candidates.csv")
print("=" * 60)
