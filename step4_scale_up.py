# Step 4: Scale up the bias test to many more genes
# Run:  python step4_scale_up.py
# Needs in this folder: variant_summary.txt.gz, AlphaMissense_hg38.tsv.gz, gnomad_cache.json (from step 2)
#
# Why this is faster than step 2: the bias test only needs BENIGN variants, so we only ask
# gnomAD about those. Variants you already fetched in step 2 are reused from the cache.
#
# TIP: leave TEST_N = 20 for a first run (checks gnomAD works, takes a minute), then set None.
# The full run can take an hour or more. You can stop it with Ctrl+C and run it again:
# progress is saved.

import gzip, hashlib, json, os, re, time
import urllib.request, urllib.error
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

# ---------------------------------------------------------------- settings
CLINVAR = "variant_summary.txt.gz"
AM_FILE = "AlphaMissense_hg38.tsv.gz"
CACHE = "gnomad_cache.json"
N_GENES = 60                  # how many genes to use, picked by number of benign variants
MIN_PER_CLASS = 20            # a gene needs at least this many benign AND pathogenic variants
MUST_INCLUDE = ["HBB", "HBA1", "HBA2", "G6PD", "APOL1", "PKLR"]   # African-relevant genes, added if present
TEST_N = None                   # set to None for the full run
SLEEP = 0.7                   # seconds between gnomAD requests
GNOMAD_URL = "https://gnomad.broadinstitute.org/api"
DATASET = "gnomad_r4"
AM_PATH = 0.564               # AlphaMissense: above this = "likely pathogenic"
AM_BENIGN = 0.34              # AlphaMissense: below this = "likely benign"
RATIOS = [2, 5, 10]           # how strict "African-enriched" is (African freq >= ratio x European freq)


# ---------------------------------------------------------------- ClinVar
def load_clinvar():
    use = ["Assembly", "GeneSymbol", "Name", "ClinicalSignificance", "Chromosome",
           "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    df = pd.read_csv(CLINVAR, sep="\t", low_memory=False, usecols=use)
    df = df[(df["Assembly"] == "GRCh38") & ~df["GeneSymbol"].astype(str).str.contains(";")]

    m = df["Name"].str.extract(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)")
    df = df.assign(ref3=m[0], pos=pd.to_numeric(m[1]), alt3=m[2]).dropna(subset=["ref3", "pos", "alt3"])
    t = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
         "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}
    df["ref"], df["alt"] = df["ref3"].map(t), df["alt3"].map(t)
    df = df.dropna(subset=["ref", "alt"])

    sig = df["ClinicalSignificance"].str.lower()
    df["y"] = np.where(sig.isin(["pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"]), 1,
              np.where(sig.isin(["benign", "likely benign", "benign/likely benign"]), 0, np.nan))
    df = df.dropna(subset=["y"]).drop_duplicates(subset=["GeneSymbol", "pos", "ref", "alt"])
    df["y"] = df["y"].astype(int)

    ok = df["ReferenceAlleleVCF"].str.fullmatch("[ACGT]+", na=False) & df["AlternateAlleleVCF"].str.fullmatch("[ACGT]+", na=False)
    df = df[ok].copy()
    df["vid"] = (df["Chromosome"].astype(str) + "-" + df["PositionVCF"].astype(int).astype(str) + "-"
                 + df["ReferenceAlleleVCF"] + "-" + df["AlternateAlleleVCF"])
    return df.reset_index(drop=True)


def select_genes(df):
    c = df.groupby("GeneSymbol")["y"].agg(n_benign=lambda s: int((s == 0).sum()), n_path=lambda s: int((s == 1).sum()))
    ok = c[(c.n_benign >= MIN_PER_CLASS) & (c.n_path >= MIN_PER_CLASS)].sort_values("n_benign", ascending=False)
    genes = list(ok.head(N_GENES).index)
    for g in MUST_INCLUDE:
        if g in c.index and g not in genes:
            genes.append(g)
    print(f"Selected {len(genes)} genes. Largest by benign variants:")
    print(c.loc[genes].sort_values("n_benign", ascending=False).head(10).to_string())
    missing = [g for g in MUST_INCLUDE if g not in c.index]
    if missing:
        print("Not in ClinVar with usable missense variants:", missing)
    return genes


# ---------------------------------------------------------------- gnomAD
QUERY = """query($id: String!) { variant(variantId: $id, dataset: %s) {
  exome { ac an populations { id ac an } }
  genome { ac an populations { id ac an } } } }""" % DATASET


def ask_gnomad(vid):
    body = json.dumps({"query": QUERY, "variables": {"id": vid}}).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "student-research-project"}
    for attempt in range(5):
        try:
            req = urllib.request.Request(GNOMAD_URL, body, headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            time.sleep(10 * (attempt + 1) if e.code == 429 else 3)
        except Exception:
            time.sleep(5)
    return None


def parse_gnomad(resp):
    v = (resp.get("data") or {}).get("variant")
    if not v:
        return {"found": False}
    pops = {}
    for part in (v.get("exome"), v.get("genome")):
        if not part:
            continue
        for p in part["populations"]:
            if "_" in p["id"]:
                continue
            a = pops.setdefault(p["id"], [0, 0])
            a[0] += p["ac"] or 0
            a[1] += p["an"] or 0
    return {"found": True, "pops": pops}


def fetch_all(vids, cache):
    todo = [v for v in vids if v not in cache]
    if TEST_N:
        todo = todo[:TEST_N]
    print(f"\n{len(vids)} benign variants needed, {len(vids) - len([v for v in vids if v not in cache])} already cached, "
          f"{len(todo)} to fetch (about {len(todo) * (SLEEP + 0.3) / 60:.0f} min)")
    try:
        for i, vid in enumerate(todo, 1):
            resp = ask_gnomad(vid)
            if resp is not None:
                cache[vid] = parse_gnomad(resp)
            if i % 50 == 0:
                json.dump(cache, open(CACHE, "w"))
                print(f"  {i}/{len(todo)} done")
            time.sleep(SLEEP)
    finally:
        json.dump(cache, open(CACHE, "w"))   # saves progress even if you press Ctrl+C


def pop_af(c, pop):
    if not c or not c.get("found") or pop not in c["pops"]:
        return 0.0, 0
    ac, an = c["pops"][pop]
    return (ac / an if an else 0.0), ac


# ---------------------------------------------------------------- AlphaMissense
def alphamissense_scores(keys):
    keys = set(keys)
    tag = hashlib.md5("|".join(sorted(keys)).encode()).hexdigest()[:8]
    subset = f"alphamissense_subset_{tag}.csv"
    if not os.path.exists(subset):
        all_cols = ["CHROM", "POS", "REF", "ALT", "genome", "uniprot_id", "transcript_id",
                    "protein_variant", "am_pathogenicity", "am_class"]
        keep = ["CHROM", "POS", "REF", "ALT", "am_pathogenicity", "am_class"]
        with gzip.open(AM_FILE, "rt") as fh:
            skip = 0
            for line in fh:
                if not line.startswith("#"):
                    break
                skip += 1
        print(f"\nScanning AlphaMissense (several minutes)... skipping {skip} comment lines")
        hits, n = [], 0
        for chunk in pd.read_csv(AM_FILE, sep="\t", header=None, skiprows=skip, names=all_cols,
                                 usecols=keep, dtype=str, chunksize=2_000_000):
            key = (chunk["CHROM"].str.replace("^chr", "", regex=True) + "-" + chunk["POS"].str.strip()
                   + "-" + chunk["REF"].str.strip() + "-" + chunk["ALT"].str.strip())
            sel = key.isin(keys)
            hits.append(chunk[sel].assign(key=key[sel]))
            n += len(chunk)
            if (n // 2_000_000) % 10 == 0:
                print(f"  scanned {n / 1e6:.0f}M rows, matched {sum(len(h) for h in hits)}")
        found = pd.concat(hits)
        if len(found) == 0:
            raise SystemExit("No AlphaMissense matches. Send me the output so far.")
        found.to_csv(subset, index=False)
    am = pd.read_csv(subset, dtype=str)
    am["am_score"] = pd.to_numeric(am["am_pathogenicity"], errors="coerce")
    return am.groupby("key")["am_score"].mean()


# ---------------------------------------------------------------- the analysis
def rate(x):
    return f"{int(x.sum())}/{len(x)} ({x.mean():.0%})" if len(x) else "none"


def analyze(b):
    """b: benign variants seen in gnomAD, with af_afr, af_nfe, ac_afr, am_score, GeneSymbol, Name."""
    b = b.copy()
    b["fp"] = b["am_score"] > AM_PATH             # wrongly called pathogenic
    b["not_clear"] = b["am_score"] >= AM_BENIGN   # not confidently benign (false alarm or "ambiguous")
    print(f"\n{len(b)} benign variants seen in gnomAD, with an AlphaMissense score, across {b['GeneSymbol'].nunique()} genes")

    print("\n" + "=" * 70)
    print("HOW OFTEN IS A BENIGN VARIANT WRONGLY FLAGGED? (by how strict 'African-enriched' is)")
    print("=" * 70)
    rows = []
    for r in RATIOS:
        enr = (b["ac_afr"] >= 2) & (b["af_afr"] >= r * b["af_nfe"])
        a, o = b["fp"][enr], b["fp"][~enr]
        a2, o2 = b["not_clear"][enr], b["not_clear"][~enr]
        p1 = fisher_exact([[a.sum(), len(a) - a.sum()], [o.sum(), len(o) - o.sum()]])[1] if len(a) and len(o) else np.nan
        p2 = fisher_exact([[a2.sum(), len(a2) - a2.sum()], [o2.sum(), len(o2) - o2.sum()]])[1] if len(a2) and len(o2) else np.nan
        rows.append({"enriched_if": f">= {r}x Europe", "n_enriched": len(a),
                     "called_pathogenic_enriched": rate(a), "called_pathogenic_other": rate(o), "p": round(p1, 3),
                     "not_clearly_benign_enriched": rate(a2), "not_clearly_benign_other": rate(o2), "p2": round(p2, 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\ncalled_pathogenic = score above 0.564.  not_clearly_benign = score of 0.34 or higher.")

    # main definition: 2x
    enr = (b["ac_afr"] >= 2) & (b["af_afr"] >= RATIOS[0] * b["af_nfe"])
    b["enr"] = enr
    a, o = b["fp"][enr].astype(float).values, b["fp"][~enr].astype(float).values
    print("\n" + "=" * 70)
    print(f"MAIN COMPARISON (enriched = African freq >= {RATIOS[0]}x European freq)")
    print("=" * 70)
    if len(a) and len(o):
        rng = np.random.default_rng(0)
        diffs = [rng.choice(a, len(a)).mean() - rng.choice(o, len(o)).mean() for _ in range(2000)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"Difference in false-alarm rate (enriched minus other): {a.mean() - o.mean():+.1%}  (95% range {lo:+.1%} to {hi:+.1%})")
        print("If that range includes 0, there is no clear difference. If it is entirely above 0, the model over-calls enriched variants.")
        u = mannwhitneyu(b["am_score"][enr], b["am_score"][~enr], alternative="two-sided")
        print(f"Average score: enriched {b['am_score'][enr].mean():.3f} vs other {b['am_score'][~enr].mean():.3f}  (Mann-Whitney p = {u.pvalue:.3f})")
    else:
        print("One of the groups is empty, so no comparison is possible.")

    # control for gene
    try:
        from statsmodels.stats.contingency_tables import StratifiedTable
        tables = []
        for g, sub in b.groupby("GeneSymbol"):
            e, x = sub["fp"][sub["enr"]], sub["fp"][~sub["enr"]]
            if len(e) and len(x):
                tables.append(np.array([[e.sum(), len(e) - e.sum()], [x.sum(), len(x) - x.sum()]]))
        st = StratifiedTable(tables)
        lo, hi = st.oddsratio_pooled_confint()
        print(f"\nSame comparison controlling for gene ({len(tables)} genes): odds ratio {st.oddsratio_pooled:.2f} "
              f"(95% range {lo:.2f} to {hi:.2f}), p = {st.test_null_odds().pvalue:.3f}")
        print("Odds ratio of 1 = no difference. Above 1 = enriched variants flagged more often.")
    except Exception as e:
        print("\n(Gene-controlled test skipped:", type(e).__name__, "- pip install statsmodels to enable)")

    # per gene
    g = b.groupby("GeneSymbol").apply(lambda s: pd.Series({
        "n_enriched": int(s["enr"].sum()), "flagged_enriched": int(s["fp"][s["enr"]].sum()),
        "n_other": int((~s["enr"]).sum()), "flagged_other": int(s["fp"][~s["enr"]].sum())}), include_groups=False)
    g = g[g["n_enriched"] >= 3].sort_values("n_enriched", ascending=False)
    if len(g):
        print("\nGenes with at least 3 African-enriched benign variants:")
        print(g.to_string())

    miss = b[b["enr"] & b["fp"]].sort_values("am_score", ascending=False)
    print(f"\nAfrican-enriched BENIGN variants called pathogenic ({len(miss)}):")
    if len(miss):
        print(miss[["GeneSymbol", "Name", "af_afr", "af_nfe", "am_score"]].head(25).to_string(index=False, max_colwidth=45))
    return b


def main():
    df = load_clinvar()
    print(f"{len(df)} labeled missense variants in ClinVar")
    genes = select_genes(df)
    benign = df[(df["y"] == 0) & df["GeneSymbol"].isin(genes)].copy()

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    fetch_all(benign["vid"].tolist(), cache)
    if TEST_N:
        print("\nTEST RUN. Sample of what came back:")
        for v in benign["vid"][:TEST_N]:
            c = cache.get(v)
            print(" ", v, "->", "not fetched" if c is None else ("in gnomAD" if c["found"] else "not in gnomAD"))
        print("\nIf some say 'in gnomAD', it works. Set TEST_N = None at the top and run again.")
        return

    benign["in_gnomad"] = [bool(cache.get(v, {}).get("found")) for v in benign["vid"]]
    benign["af_afr"], benign["ac_afr"] = zip(*[pop_af(cache.get(v), "afr") for v in benign["vid"]])
    benign["af_nfe"], _ = zip(*[pop_af(cache.get(v), "nfe") for v in benign["vid"]])
    print(f"\n{benign['in_gnomad'].sum()} of {len(benign)} benign variants found in gnomAD")

    seen = benign[benign["in_gnomad"]].copy()
    scores = alphamissense_scores(seen["vid"])
    seen["am_score"] = seen["vid"].map(scores)
    seen = seen[seen["am_score"].notna()]
    out = analyze(seen)
    out.drop(columns=["ref3", "alt3"], errors="ignore").to_csv("step4_results.csv", index=False)
    print("\nSaved step4_results.csv")


if __name__ == "__main__":
    main()
