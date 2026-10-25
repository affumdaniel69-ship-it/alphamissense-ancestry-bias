# Step 1: ClinVar missense baseline (pathogenic vs benign)
# Run in Jupyter/VS Code (each "# %%" is a cell) or just: python step1_baseline.py
# Setup once:  pip install pandas scikit-learn biopython
# Data once:   download variant_summary.txt.gz from
#   https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
#   and put it in the same folder as this file.

# %% 1. Settings
import re
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Start with HBB only to learn the pipeline. HBB has few benign variants, so
# once it runs end to end, add more genes below to get enough data.
GENES = ["HBB"]
GENES = ["HBB", "HBA1", "HBA2", "G6PD", "CFTR", "MYH7", "LDLR", "BRCA2"]

# %% 2. Load ClinVar and keep clean missense variants
df = pd.read_csv("variant_summary.txt.gz", sep="\t", low_memory=False)
df = df[(df["Assembly"] == "GRCh38") & (df["GeneSymbol"].isin(GENES))]

# Name looks like: NM_000518.5(HBB):c.20A>T (p.Glu7Val)
pat = re.compile(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)")
m = df["Name"].str.extract(pat)
df = df.assign(ref3=m[0], pos=pd.to_numeric(m[1]), alt3=m[2]).dropna(subset=["ref3", "pos", "alt3"])

three2one = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
             "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}
df["ref"] = df["ref3"].map(three2one)
df["alt"] = df["alt3"].map(three2one)
df = df.dropna(subset=["ref", "alt"])  # drops stops, "Ter", etc.

# Labels: only confident calls. Uncertain/conflicting variants are dropped.
sig = df["ClinicalSignificance"].str.lower()
df["y"] = np.where(sig.isin(["pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"]), 1,
          np.where(sig.isin(["benign", "likely benign", "benign/likely benign"]), 0, np.nan))
df = df.dropna(subset=["y"]).drop_duplicates(subset=["GeneSymbol", "pos", "ref", "alt"])
df["y"] = df["y"].astype(int)
print(df.groupby("GeneSymbol")["y"].value_counts().unstack(fill_value=0))

# %% 3. Features: how different are the two amino acids?
blosum = substitution_matrices.load("BLOSUM62")
hydro = dict(zip("ARNDCQEGHILKMFPSTWYV",  # Kyte-Doolittle hydrophobicity
    [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,-0.9,-1.3,4.2]))
volume = dict(zip("ARNDCQEGHILKMFPSTWYV",  # side-chain volume, cubic angstroms
    [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
charge = {"D":-1,"E":-1,"K":1,"R":1,"H":0.5}

df["blosum"] = [blosum[r][a] for r, a in zip(df.ref, df.alt)]
df["d_hydro"] = df.alt.map(hydro) - df.ref.map(hydro)
df["d_volume"] = df.alt.map(volume) - df.ref.map(volume)
df["d_charge"] = df.alt.map(lambda a: charge.get(a, 0)) - df.ref.map(lambda a: charge.get(a, 0))
df["to_proline"] = (df.alt == "P").astype(int)
df["from_glycine"] = (df.ref == "G").astype(int)
FEATURES = ["blosum", "d_hydro", "d_volume", "d_charge", "to_proline", "from_glycine"]

# %% 4. Split by position so the model can't memorise a site, then train
X, y, groups = df[FEATURES], df["y"], df["GeneSymbol"] + df["pos"].astype(str)
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(X, y, groups))

models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "grad_boost": GradientBoostingClassifier(random_state=0),
}
for name, model in models.items():
    model.fit(X.iloc[tr], y.iloc[tr])
    p = model.predict_proba(X.iloc[te])[:, 1]
    print(f"\n{name}: AUC = {roc_auc_score(y.iloc[te], p):.3f}")
    print(classification_report(y.iloc[te], p > 0.5, zero_division=0))
