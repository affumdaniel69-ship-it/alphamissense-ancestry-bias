# Step 5: Is the "African-enriched variants get flagged less" result just a frequency effect?
# Run:  python step5_frequency_matched.py
# Needs in this folder: step4_results.csv and gnomad_cache.json (both made by step 4). No downloads.
#
# Idea: compare enriched vs other variants that have the SAME overall frequency (and the same gene).
# If the gap in false-alarm rate disappears, the earlier result was about how common a variant is,
# not about ancestry. If it survives, it is a stronger finding.

import json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable

AM_PATH = 0.564     # AlphaMissense: above this = "likely pathogenic"
ENRICH_RATIO = 2    # African freq >= 2x European freq, and at least 2 African alleles
# Bands of how many times the variant was seen in gnomAD (all populations). Fine at the low end, because
# very rare variants differ a lot from each other and a coarse band would leave a hidden imbalance.
BINS = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500, 1000, 5000, 20000, 100000, 1e9]


def total_counts(cache, vid):
    c = cache.get(vid)
    if not c or not c.get("found"):
        return 0, 0
    # gnomAD also lists whole-sex totals ("XX", "XY") alongside the ancestry groups.
    # Adding those in would count every allele twice, so leave them out.
    pops = {k: v for k, v in c["pops"].items() if k not in ("XX", "XY")}
    ac = sum(v[0] for v in pops.values())
    an = sum(v[1] for v in pops.values())
    return ac, an


def pct(x):
    return f"{int(x.sum())}/{len(x)} ({x.mean():.1%})" if len(x) else "none"


def pooled_or(df, strata_cols):
    """Mantel-Haenszel odds ratio of being flagged, enriched vs other, within strata."""
    tables = []
    for _, s in df.groupby(strata_cols, observed=True):
        e, o = s["fp"][s["enr"]], s["fp"][~s["enr"]]
        if len(e) and len(o):
            tables.append(np.array([[e.sum(), len(e) - e.sum()], [o.sum(), len(o) - o.sum()]]))
    if not tables:
        return None
    st = StratifiedTable(tables)
    lo, hi = st.oddsratio_pooled_confint()
    return st.oddsratio_pooled, lo, hi, st.test_null_odds().pvalue, len(tables)


def report(df, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{len(df)} benign variants ({int(df['enr'].sum())} African-enriched)")
    e, o = df["fp"][df["enr"]], df["fp"][~df["enr"]]
    print(f"Raw:  enriched {pct(e)}   other {pct(o)}")

    rows = []
    for b, s in df.groupby("af_bin", observed=True):
        se, so = s["fp"][s["enr"]], s["fp"][~s["enr"]]
        rows.append({"times_seen_in_gnomad": str(b), "n_enriched": len(se), "n_other": len(so),
                     "flagged_enriched": f"{se.mean():.0%}" if len(se) else "-",
                     "flagged_other": f"{so.mean():.0%}" if len(so) else "-",
                     "mean_total_AF_enriched": f"{s['total_af'][s['enr']].mean():.1e}" if len(se) else "-",
                     "mean_total_AF_other": f"{s['total_af'][~s['enr']].mean():.1e}" if len(so) else "-"})
    print("\nBy how often the variant is seen overall (this is the matched comparison):")
    print(pd.DataFrame(rows).to_string(index=False))

    r1 = pooled_or(df, ["af_bin"])
    r2 = pooled_or(df, ["af_bin", "GeneSymbol"])
    print("\nOdds ratio of being flagged, enriched vs other (1.0 = no difference):")
    if r1:
        print(f"  matched on frequency:          {r1[0]:.2f}  (95% range {r1[1]:.2f} to {r1[2]:.2f}), p = {r1[3]:.3f}")
    if r2:
        print(f"  matched on frequency AND gene: {r2[0]:.2f}  (95% range {r2[1]:.2f} to {r2[2]:.2f}), p = {r2[3]:.3f}  ({r2[4]} strata)")
    return r2 or r1


def main():
    df = pd.read_csv("step4_results.csv")
    cache = json.load(open("gnomad_cache.json"))

    tc = [total_counts(cache, v) for v in df["vid"]]
    df["total_ac"] = [t[0] for t in tc]
    df["total_an"] = [t[1] for t in tc]
    df = df[(df["total_an"] > 0) & (df["total_ac"] >= 1)].copy()
    df["total_af"] = df["total_ac"] / df["total_an"]

    # recompute the flags so this script does not depend on how step 4 saved them
    df["enr"] = (df["ac_afr"] >= 2) & (df["af_afr"] >= ENRICH_RATIO * df["af_nfe"])
    df["fp"] = df["am_score"] > AM_PATH
    df["af_bin"] = pd.cut(df["total_ac"], BINS)

    print(f"{len(df)} benign variants with a usable overall frequency")
    print("\nHow different are the two groups in overall frequency? (this is the thing we must control for)")
    print(df.groupby("enr")["total_af"].describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]]
          .rename(index={False: "other", True: "african_enriched"}).to_string(float_format=lambda x: f"{x:.1e}"))

    r_all = report(df, "ALL VARIANTS")
    r_ac = report(df[df["total_ac"] >= 10], "ONLY VARIANTS SEEN 10+ TIMES IN GNOMAD (removes the rarest ones)")

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("Step 4 found an odds ratio of about 0.68 (enriched variants flagged less often).")
    print("- If the matched odds ratios above are still clearly below 1 (range does not include 1),")
    print("  the result survives: AlphaMissense does not over-call African-enriched benign variants,")
    print("  even compared with equally common variants.")
    print("- If they move to about 1 (range includes 1), the earlier gap was mostly because enriched")
    print("  variants are more common overall. Then the honest finding is: no ancestry-specific bias")
    print("  detected, and frequency explains the rest.")
    print("- If they go above 1, tell me, because that would be a bias signal hiding behind frequency.")
    df.to_csv("step5_results.csv", index=False)
    print("\nSaved step5_results.csv")


if __name__ == "__main__":
    main()
