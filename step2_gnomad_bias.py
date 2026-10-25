# Step 2: Does the model fail more often on variants enriched in African-ancestry populations?
# Run:  python step2_gnomad_bias.py
# Needs step 1 packages + scipy (comes with scikit-learn). Needs internet (talks to gnomAD).
# First run takes a while (about 1 request per variant). Results are saved to
# gnomad_cache.json, so if it stops you can just run it again and it continues.
#
# TIP: set TEST_N = 20 first to check that the gnomAD part works, then set it to None.

# %% 1. Settings
import json, os, re, time
import urllib.request, urllib.error
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

GENES = ["HBB", "HBA1", "HBA2", "G6PD", "CFTR", "MYH7", "LDLR", "BRCA2"]
TEST_N = None            # set to None for the full run
SLEEP = 0.7            # seconds between gnomAD requests (be polite, avoid rate limits)
CACHE = "gnomad_cache.json"
GNOMAD_URL = "https://gnomad.broadinstitute.org/api"
DATASET = "gnomad_r4"  # GRCh38

# %% 2. Load ClinVar (same cleaning as step 1, plus genomic coordinates)
df = pd.read_csv("variant_summary.txt.gz", sep="\t", low_memory=False)
df = df[(df["Assembly"] == "GRCh38") & (df["GeneSymbol"].isin(GENES))]

pat = re.compile(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)")
m = df["Name"].str.extract(pat)
df = df.assign(ref3=m[0], pos=pd.to_numeric(m[1]), alt3=m[2]).dropna(subset=["ref3", "pos", "alt3"])

three2one = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
             "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}
df["ref"] = df["ref3"].map(three2one)
df["alt"] = df["alt3"].map(three2one)
df = df.dropna(subset=["ref", "alt"])

sig = df["ClinicalSignificance"].str.lower()
df["y"] = np.where(sig.isin(["pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"]), 1,
          np.where(sig.isin(["benign", "likely benign", "benign/likely benign"]), 0, np.nan))
df = df.dropna(subset=["y"]).drop_duplicates(subset=["GeneSymbol", "pos", "ref", "alt"])
df["y"] = df["y"].astype(int)

# gnomAD variant id looks like 11-5227002-T-A
df = df[df["ReferenceAlleleVCF"].str.fullmatch("[ACGT]+", na=False) & df["AlternateAlleleVCF"].str.fullmatch("[ACGT]+", na=False)]
df["vid"] = (df["Chromosome"].astype(str) + "-" + df["PositionVCF"].astype(int).astype(str) + "-"
             + df["ReferenceAlleleVCF"] + "-" + df["AlternateAlleleVCF"])
df = df.reset_index(drop=True)
print(f"{len(df)} variants with labels and genomic coordinates")

# %% 3. Fetch population frequencies from gnomAD (cached to disk)
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
    return None  # failed: not cached, so a re-run will try again

def parse(resp):
    v = (resp.get("data") or {}).get("variant")
    if not v:
        return {"found": False}
    pops = {}
    for part in (v.get("exome"), v.get("genome")):
        if not part:
            continue
        for p in part["populations"]:
            if "_" in p["id"]:      # skip sex sub-groups like afr_XX
                continue
            a = pops.setdefault(p["id"], [0, 0])
            a[0] += p["ac"] or 0
            a[1] += p["an"] or 0
    return {"found": True, "pops": pops}

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
todo = df["vid"].tolist()
if TEST_N:
    todo = todo[:TEST_N]
todo = [v for v in todo if v not in cache]
print(f"{len(todo)} variants to fetch from gnomAD (about {len(todo) * SLEEP / 60:.0f} min)")

for i, vid in enumerate(todo, 1):
    resp = ask_gnomad(vid)
    if resp is not None:
        cache[vid] = parse(resp)
    if i % 50 == 0:
        json.dump(cache, open(CACHE, "w"))
        print(f"  {i}/{len(todo)} done")
    time.sleep(SLEEP)
json.dump(cache, open(CACHE, "w"))

if TEST_N:
    print("\nTEST RUN ONLY. Here is what came back:")
    for v in df["vid"][:TEST_N]:
        c = cache.get(v)
        print(v, "->", "not fetched" if c is None else ("in gnomAD" if c["found"] else "not in gnomAD"))
    print("\nIf you see 'in gnomAD' for some, it works. Set TEST_N = None and run again.")
    raise SystemExit

# %% 4. Turn gnomAD results into columns
def af(c, pop):
    if not c or not c["found"] or pop not in c["pops"]:
        return 0.0, 0
    ac, an = c["pops"][pop]
    return (ac / an if an else 0.0), ac

df["in_gnomad"] = [bool(cache.get(v, {}).get("found")) for v in df["vid"]]
df["af_afr"], df["ac_afr"] = zip(*[af(cache.get(v), "afr") for v in df["vid"]])
df["af_nfe"], _ = zip(*[af(cache.get(v), "nfe") for v in df["vid"]])  # non-Finnish European

# "African-enriched": seen in African-ancestry samples at least twice as often as in Europeans
df["afr_enriched"] = (df["ac_afr"] >= 2) & (df["af_afr"] >= 2 * df["af_nfe"])

print("\nVariants found in gnomAD, by ClinVar label (0 = benign, 1 = pathogenic):")
print(pd.crosstab(df["in_gnomad"], df["y"]))
print("\nAfrican-enriched variants, by label:")
print(pd.crosstab(df["afr_enriched"], df["y"]))

# %% 5. Features (same as step 1)
blosum = substitution_matrices.load("BLOSUM62")
hydro = dict(zip("ARNDCQEGHILKMFPSTWYV",
    [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,-0.9,-1.3,4.2]))
volume = dict(zip("ARNDCQEGHILKMFPSTWYV",
    [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
charge = {"D":-1,"E":-1,"K":1,"R":1,"H":0.5}

df["blosum"] = [blosum[r][a] for r, a in zip(df.ref, df.alt)]
df["d_hydro"] = df.alt.map(hydro) - df.ref.map(hydro)
df["d_volume"] = df.alt.map(volume) - df.ref.map(volume)
df["d_charge"] = df.alt.map(lambda a: charge.get(a, 0)) - df.ref.map(lambda a: charge.get(a, 0))
df["to_proline"] = (df.alt == "P").astype(int)
df["from_glycine"] = (df.ref == "G").astype(int)
# NOTE: allele frequency is deliberately NOT a feature. ClinVar calls variants benign
# partly because they are common, so using it would leak the answer.
FEATURES = ["blosum", "d_hydro", "d_volume", "d_charge", "to_proline", "from_glycine"]

# %% 6. Out-of-fold predictions: every variant gets scored by a model that never saw its position
X, y = df[FEATURES], df["y"]
groups = df["GeneSymbol"] + df["pos"].astype(str)

models = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "grad_boost": lambda: GradientBoostingClassifier(random_state=0),
}
for name, make in models.items():
    p = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        mdl = make().fit(X.iloc[tr], y.iloc[tr])
        p[te] = mdl.predict_proba(X.iloc[te])[:, 1]
    df[f"p_{name}"] = p
    df[f"wrong_{name}"] = ((p > 0.5).astype(int) != y).astype(int)
    print(f"\n{name}: overall AUC = {roc_auc_score(y, p):.3f}")

# %% 7. The bias test
# Focus: benign variants (y = 0). If the model calls an African-enriched benign variant
# "pathogenic" more often than other benign variants, that is the bias we are looking for.
print("\n" + "=" * 60)
print("BIAS TEST: false-alarm rate on BENIGN variants seen in gnomAD")
print("=" * 60)
benign = df[(df["y"] == 0) & df["in_gnomad"]]
for name in models:
    fp = (benign[f"p_{name}"] > 0.5)
    a, b = fp[benign["afr_enriched"]], fp[~benign["afr_enriched"]]
    print(f"\n{name}")
    print(f"  African-enriched:     {a.sum()}/{len(a)} wrongly called pathogenic ({a.mean():.0%})" if len(a) else "  African-enriched: no variants")
    print(f"  Not African-enriched: {b.sum()}/{len(b)} wrongly called pathogenic ({b.mean():.0%})" if len(b) else "  Not enriched: no variants")
    if len(a) and len(b):
        _, pval = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
        print(f"  Fisher exact p-value: {pval:.3f}  (below 0.05 = difference unlikely to be chance)")

print("\nCaveat: if the African-enriched group has fewer than ~20 variants, treat any")
print("difference as a hint, not a result.")

# %% 8. Save everything for step 3
df.drop(columns=["ref3", "alt3"]).to_csv("step2_results.csv", index=False)
print("\nSaved step2_results.csv")
