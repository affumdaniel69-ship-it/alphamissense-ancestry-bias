# Step 3: Test AlphaMissense on the same variants and rerun the bias test
# Needs: step2_results.csv (from step 2) and AlphaMissense_hg38.tsv.gz in this folder.
# Download (643 MB, do NOT unzip):
#   https://zenodo.org/records/10813168/files/AlphaMissense_hg38.tsv.gz?download=1
# Run:  python step3_alphamissense.py
# First run takes several minutes (it scans ~200 million rows once). It saves a small
# filtered file, so later runs are instant.

# %% 1. Load step 2 results
import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

df = pd.read_csv("step2_results.csv")
df["key"] = df["vid"].astype(str).str.replace("^chr", "", regex=True)   # e.g. 11-5227002-T-A
wanted = set(df["key"])
print(f"{len(df)} variants from step 2")
print("Example keys we are looking for:", list(df["key"].head(3)))

# %% 2. Pull only our variants out of the huge AlphaMissense file
SUBSET = "alphamissense_subset_v2.csv"   # new name: the old file had empty score columns
if os.path.exists(SUBSET) and os.path.getsize(SUBSET) < 200:
    os.remove(SUBSET)   # an empty file left over from a failed run: scan again
if not os.path.exists(SUBSET):
    import gzip
    all_cols = ["CHROM", "POS", "REF", "ALT", "genome", "uniprot_id", "transcript_id",
                "protein_variant", "am_pathogenicity", "am_class"]
    keep = ["CHROM", "POS", "REF", "ALT", "am_pathogenicity", "am_class"]
    # The file starts with comment lines (#...). Count them so pandas can skip them cleanly;
    # otherwise it miscounts the columns and the score columns come back empty.
    with gzip.open("AlphaMissense_hg38.tsv.gz", "rt") as fh:
        skip = 0
        for line in fh:
            if not line.startswith("#"):
                break
            skip += 1
    print(f"Skipping {skip} header/comment lines")
    hits, n = [], 0
    reader = pd.read_csv("AlphaMissense_hg38.tsv.gz", sep="\t", header=None, skiprows=skip,
                         names=all_cols, usecols=keep, dtype=str, chunksize=2_000_000)
    for chunk in reader:
        chunk = chunk[~chunk["CHROM"].str.startswith("#")]   # drop the header/comment lines
        if n == 0:
            print("First rows of the AlphaMissense file, as we read them:")
            print(chunk.head(3).to_string(index=False))
        chrom = chunk["CHROM"].str.replace("^chr", "", regex=True)
        key = chrom + "-" + chunk["POS"].str.strip() + "-" + chunk["REF"].str.strip() + "-" + chunk["ALT"].str.strip()
        sel = key.isin(wanted)
        hits.append(chunk[sel].assign(key=key[sel]))
        n += len(chunk)
        print(f"  scanned {n/1e6:.0f}M rows, matched {sum(len(h) for h in hits)}")
    found = pd.concat(hits)
    if len(found) == 0:
        raise SystemExit("\nNo variants matched. Send me the 'First rows' and 'Example keys' lines above.")
    found.to_csv(SUBSET, index=False)
am = pd.read_csv(SUBSET, dtype=str)
print(f"\nAlphaMissense subset file has {len(am)} rows")
# rebuild the key from the columns, in case an older run saved it in a different format
am["key"] = (am["CHROM"].str.replace("^chr", "", regex=True) + "-" + am["POS"].str.strip()
             + "-" + am["REF"].str.strip() + "-" + am["ALT"].str.strip())
print("Example keys in subset:", list(am["key"].head(3)))
print("Overlap with our keys:", len(set(am["key"]) & wanted))
am["am_pathogenicity"] = pd.to_numeric(am["am_pathogenicity"], errors="coerce")
print("Example scores:", list(am["am_pathogenicity"].head(3)))
if am["am_pathogenicity"].notna().sum() == 0:
    os.remove(SUBSET)
    raise SystemExit("\nThe score column is empty. I deleted the subset file; send me the first 15 lines printed.")
am = am.groupby("key", as_index=False).agg(am_score=("am_pathogenicity", "mean"))
df = df.merge(am, on="key", how="left")
print(f"AlphaMissense scored {df['am_score'].notna().sum()} of {len(df)} variants")
if df["am_score"].notna().sum() == 0:
    raise SystemExit("\nStill zero matches. Delete alphamissense_subset.csv, run again, and send me "
                     "everything this script printed from the top.")

# %% 3. Overall accuracy: your models vs AlphaMissense
scored = df[df["am_score"].notna()]
print("\nOverall AUC on variants that AlphaMissense scored:")
for label, col in [("your logistic", "p_logistic"), ("your grad_boost", "p_grad_boost"), ("AlphaMissense", "am_score")]:
    print(f"  {label:16s} {roc_auc_score(scored['y'], scored[col]):.3f}")

# %% 4. The bias test, side by side
# AlphaMissense's own cutoff: score above 0.564 = "likely pathogenic".
AM_CUTOFF = 0.564
predictors = {"your logistic": ("p_logistic", 0.5),
              "your grad_boost": ("p_grad_boost", 0.5),
              "AlphaMissense": ("am_score", AM_CUTOFF)}

print("\n" + "=" * 64)
print("FALSE-ALARM RATE on benign variants seen in gnomAD")
print("(share of benign variants wrongly called pathogenic)")
print("=" * 64)
benign = df[(df["y"] == 0) & df["in_gnomad"] & df["am_score"].notna()]
rows = []
for label, (col, cut) in predictors.items():
    fp = benign[col] > cut
    a, b = fp[benign["afr_enriched"]], fp[~benign["afr_enriched"]]
    _, pval = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
    rows.append({"predictor": label,
                 "afr_enriched": f"{a.sum()}/{len(a)} ({a.mean():.0%})",
                 "other": f"{b.sum()}/{len(b)} ({b.mean():.0%})",
                 "p_value": round(pval, 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\nAUC inside each group (all scored variants):")
for label, (col, cut) in predictors.items():
    out = []
    for name, mask in [("afr_enriched", scored["afr_enriched"]), ("other", ~scored["afr_enriched"])]:
        sub = scored[mask]
        out.append(f"{name}: {roc_auc_score(sub['y'], sub[col]):.3f} (n={len(sub)})" if sub["y"].nunique() == 2 else f"{name}: n/a")
    print(f"  {label:16s} " + "   ".join(out))

# %% 5. Which African-enriched benign variants does AlphaMissense get wrong?
miss = benign[benign["afr_enriched"] & (benign["am_score"] > AM_CUTOFF)]
print(f"\nAfrican-enriched BENIGN variants that AlphaMissense calls pathogenic ({len(miss)}):")
if len(miss):
    show = miss[["GeneSymbol", "Name", "af_afr", "af_nfe", "am_score"]].sort_values("am_score", ascending=False)
    print(show.to_string(index=False, max_colwidth=45))

# %% 6. Save
df.to_csv("step3_results.csv", index=False)
print("\nSaved step3_results.csv")
print("\nHow to read this: a big gap in the 'afr_enriched' vs 'other' columns for AlphaMissense")
print("but not for your simple models would be your finding. No gap is a valid result too.")
