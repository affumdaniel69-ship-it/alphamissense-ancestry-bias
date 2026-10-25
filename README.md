# Does AlphaMissense over-call benign variants that are enriched in African-ancestry populations?

A small bioinformatics project testing whether a leading variant-effect predictor (AlphaMissense) wrongly flags harmless missense variants as pathogenic more often when those variants are enriched in African-ancestry populations.

**Author:** Daniel Kwabena Affum, Biomedical Engineering, KNUST

## Result in one paragraph

Across 13,480 ClinVar benign missense variants in 64 genes, AlphaMissense did **not** over-call African-enriched variants. It flagged them as pathogenic less often (7.7% vs 11.3%), and the gap persisted after matching on overall allele frequency and gene (odds ratio 0.79, 95% CI 0.67 to 0.94).

| Analysis | Variants | Enriched flagged | Other flagged | Odds ratio (95% CI) |
| --- | --- | --- | --- | --- |
| Adjusted for gene | 13,565 | 207/2,676 (7.7%) | 1,234/10,889 (11.3%) | 0.68 (0.58 to 0.80) |
| Matched on frequency and gene | 13,480 | 207/2,676 (7.7%) | 1,223/10,804 (11.3%) | 0.79 (0.67 to 0.94) |
| Same, variants seen 10+ times | 7,223 | 119/1,921 (6.2%) | 448/5,302 (8.4%) | 0.74 (0.59 to 0.93) |

An odds ratio below 1 means African-enriched variants were flagged less often. Roughly a third of the gene-adjusted gap (0.68) is explained by enriched variants being more common overall; the rest (0.79) remains.

## Method

- **Variants:** ClinVar (GRCh38) missense variants labeled benign or likely benign, from 66 selected genes (the 60 with the most benign variants, each with at least 20 benign and 20 pathogenic variants, plus HBB, HBA1, HBA2, G6PD, APOL1 and PKLR). 64 genes contributed variants.
- **Ancestry proxy:** gnomAD v4 allele counts. A variant is "African-enriched" if it has at least 2 alleles in the African/African American group and a frequency there at least 2x that in non-Finnish Europeans (2,676 variants, 19.9%).
- **Outcome:** a benign variant is a false alarm if its AlphaMissense score is above 0.564 (the model's own "likely pathogenic" cutoff).
- **Statistics:** Fisher exact tests for raw comparisons. The main estimate is a Mantel-Haenszel odds ratio within strata of overall allele count (17 bands) and gene (621 strata). Sensitivity checks: variants seen 10+ times in gnomAD, and stricter enrichment cutoffs (5x and 10x).

## Limitations

- **Labels and predictor both lean on frequency.** ClinVar reviewers often classify a variant as benign because it is common in some population, and AlphaMissense is understood to draw on population variant data in training. The enriched benign variants tested here are likely the easiest cases.
- **Unlabeled variants are not tested.** Variants of uncertain significance, where bias would most plausibly show up, have no label to score against.
- **The ancestry proxy is coarse.** Enrichment is inferred from gnomAD's African/African American group, a convenience sample, not individual ancestry.
- **One predictor, one direction.** Only AlphaMissense was tested, and only false alarms on benign variants. Missed pathogenic variants were not assessed.

## How to reproduce

Python 3 with:

```
pip install pandas scikit-learn biopython scipy statsmodels
```

Download these files into the same folder as the scripts (do not unzip):

| File | Where from |
| --- | --- |
| `variant_summary.txt.gz` | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz |
| `AlphaMissense_hg38.tsv.gz` (643 MB) | https://zenodo.org/records/10813168/files/AlphaMissense_hg38.tsv.gz?download=1 |

gnomAD frequencies are fetched automatically from the gnomAD API and cached in `gnomad_cache.json`.

Run the scripts in order:

| Script | What it does |
| --- | --- |
| `step1_baseline.py` | Pilot: trains simple models on amino-acid properties (8 genes) to get a baseline AUC (about 0.7). |
| `step2_gnomad_bias.py` | Pilot: fetches gnomAD frequencies and runs a first bias test on 8 genes. Set `TEST_N = None` for the full run. |
| `step3_alphamissense.py` | Pilot: adds AlphaMissense scores for the same 8 genes (AUC about 0.955). |
| `step4_scale_up.py` | Main analysis: 66 genes, benign variants only. Fetches up to `MAX_NEW` new variants per run, so re-run until the cache is complete (several hours in total). |
| `step5_frequency_matched.py` | Frequency- and gene-matched comparison that produces the headline odds ratios. Needs `step4_results.csv` and `gnomad_cache.json`. |

## AI use disclosure

Claude (Anthropic's AI assistant) helped plan the analysis, wrote the Python code and drafted this README and the accompanying summaries. I ran all code locally on public data, and every figure above comes from those runs. One counting error in the AI-written code (sex-specific totals in gnomAD were added into overall counts, doubling them) was found during output checks and corrected before the final matched analysis. The work has not been peer reviewed. It is a student project, not medical advice.

## Data sources and credit

- ClinVar, NCBI.
- gnomAD v4, Broad Institute.
- AlphaMissense scores, Google DeepMind, released under CC-BY 4.0 (Cheng et al., 2023, *Science*).
