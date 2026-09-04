# Causal Inference Case Study: Restaurant Rating Attribution Analysis


## Project Overview

This project applies **causal inference** techniques to a restaurant recommendation dataset to answer a core business question: **What factors causally drive overall restaurant ratings, and through which pathways do they operate?**

Rather than relying on correlations alone, this analysis constructs a formal causal graph (DAG), identifies treatment variables, estimates causal effects using DoWhy and custom pipelines, and decomposes those effects through mediation and heterogeneous treatment effect (HTE) analysis.

### Business Context

This project mirrors the analytical challenges faced by data science teams at companies  — where understanding *why* metrics move (not just *that* they move) is essential for building scalable decision-making infrastructure. The methods demonstrated here translate directly to questions like revenue attribution, user behavior modeling, and A/B test analysis in production environments.

---

## Key Analytical Questions

1. **Which factors directly cause changes in restaurant ratings?** (Causal graph construction and attribution)
2. **How much of each treatment's effect flows through food quality vs. service quality?** (Mediation analysis)
3. **Do causal effects differ across customer subgroups?** (Heterogeneous treatment effects)
4. **How does graph-informed causal attribution differ from naive correlation-based attribution?** (DoWhy counterfactual estimation vs. Shapley decomposition)

---

## Project Structure

```
Causal_inference-Case_study/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── pyproject.toml                         # Package configuration
│
├── 01_Data_Analysis/                      # Phase 1: Data pipeline
│   ├── data_pipeline.py                   # Canonical data source — get_initial_data()
│   ├── final_data.pkl                     # Cached output (1,161 rows × 52 features)
│   └── Reports/
│       └── data_pipeline_results.txt      # Pipeline run logs
│
├── 02_Causal_Graph/                       # Phase 2: Causal graph & correlation EDA
│   ├── causal_graph_module_complete.py    # DAG construction, visualization, path analysis
│   ├── correlation_analyis.py             # ydata-profiling EDA + BeautifulSoup extraction
│   └── Reports/
│       ├── causal_graph.png               # 16-edge DAG visualization (corrected)
│       └── correlation_analysis.txt       # Graph construction summary
│
├── 03_Causal_ATTRIBUTIONAnalysis/         # Phase 3: Dual attribution analysis
│   ├── Attribution_analysis_module.py     # Shapley values, R², OLS relative importance
│   ├── causal_graph_attribution_module.py # DoWhy backdoor identification (formal causal)
│   └── Reports/                           # Output charts and results (auto-created)
│
├── 04_Mediation_HTE/                      # Phase 4: Mediation analysis & HTE
│   ├── mediation_analysis.py              # Baron & Kenny mediation (MediationAnalysis class)
│   ├── HTE_analysis.py                    # Hijos HTE — IPW, propensity scores, 17 pathways
│   └── results/                           # Output charts, CSVs, and text reports
│       ├── mediation_results.png
│       ├── mediation_summary.csv
│       ├── mediation_analysis_results.txt
│       ├── HTE_analysis_summary.png
│       └── HTE_analysis_results.txt
│
└── causal_case_study_interview_prep/      # Archived exploratory work
    ├── data/raw/                          # 9 source CSV files
    └── Reports/                           # Earlier output artifacts
```

---

## Data Pipeline (Phase 1)

All downstream analysis imports data exclusively via `get_initial_data()` from `01_Data_Analysis/data_pipeline.py`.

### 1. Data Ingestion & Merging

Nine relational CSV files are loaded and merged using `pandas.merge()` with left joins:

- **User data**: `userprofile` → left merge `userpayment` → left merge `usercuisine` → left merge `rating_final`
- **Restaurant data**: `geoplaces2` → left merge `chefmozaccepts` → left merge `chefmozparking` → left merge `chefmozcuisine` → left merge `chefmozhours4`
- **Combined**: User data left merge Restaurant data on `placeID`
- **Output**: 1,161 rows × 52 features (one row per rating — fan-out collapsed)

### 2. Feature Engineering

- **Patron-Restaurant Distance**: Geodesic distance (km) using the Haversine formula
- **Location Clusters**: KMeans (k=5) on restaurant latitude/longitude
- **Age Groups**: Derived from birth year → bins 18-25, 26-35, 36-50, 50+
- **Cuisine Match Score**: Jaccard similarity (0–1) between user and restaurant cuisines
- **Business Hours Categories**: Parsed into Morning, Afternoon, Evening, Full Day, 24H

### 3. Preprocessing

- Categorical cleaning (replacing `?` markers with NaN)
- Missing value imputation (most-frequent strategy for 17 categorical columns)
- Label encoding for causal model compatibility

---

## Causal Analysis Framework

### Phase 2: Causal Graph (DAG)

The causal graph was re-derived on the corrected **1,161-row** dataset (one row per rating). The previous 78-edge graph was built on the 31,559-row fan-out, which manufactured spurious associations between user demographics and rating — each user was replicated once per cuisine × payment combination, re-counting their fixed attributes against their ratings.

The corrected graph has **16 edges** across **15 nodes** and is a strict DAG (no cycles). Variables are layered so edges only run forward:

- **L0 exogenous** — user and restaurant attributes
- **L1 dyadic** — `patron_restaurant_distance`, `cuisine_match_score`
- **L2 mediators** — `food_rating`, `service_rating`
- **L3 outcome** — `rating`

The graph identifies **3 direct treatment variables** on `rating`:

1. `food_rating` — Food quality score (association 0.692)
2. `service_rating` — Service quality score (association 0.680)
3. `color` — User color preference (association 0.223)

Five variables from the original 7-treatment list (`hijos`, `personality`, `height`, `interest`, and `activity`) did not clear the 0.20 association threshold on the corrected data and are absent from the graph. `drink_level` and `Upayment` are L0 exogenous variables that reach `rating` indirectly via `food_rating` and `service_rating`.

> **Threshold note**: only `food_rating` and `service_rating` clear ydata-profiling's own 0.50 alert level. All other edges represent weak associations (≤ 0.35) retained to give the graph structure to identify against. The gap between 0.680 and 0.223 is real and should be interpreted accordingly.

### Phase 3: Dual Attribution Methods

The project uses two complementary approaches to attribution:

**Method 1 — Shapley / OLS (`Attribution_analysis_module.py`)**

- Naive correlation baseline
- Direct effects (OLS, standardized)
- Unique R² contribution (drop-one decomposition)
- Shapley value decomposition across all treatment combinations
- **Key result**: R² = 82.84%; `food_rating` (42.58%) and `service_rating` (41.72%) dominate

**Method 2 — DoWhy Backdoor (`causal_graph_attribution_module.py`)**

- Formal causal identification using the backdoor criterion
- Estimates the ATE via `CausalModel` with graph-informed confounders
- **Key result**: `food_rating` causal estimate = 0.549 vs. naive correlation 0.864 (Δ = −0.315); `service_rating` = 0.533 vs. 0.852 (Δ = −0.319). Graph-based adjustment substantially reduces naive estimates, confirming confounding.

### Phase 4: Mediation Analysis

For each treatment, effects are decomposed into:

- **Direct Effect** (c'): Treatment → Rating (controlling for mediators)
- **Indirect via Food** (a₁ × b₁): Treatment → Food Rating → Overall Rating
- **Indirect via Service** (a₂ × b₂): Treatment → Service Rating → Overall Rating

Treatments analysed are the three L0 exogenous variables with paths to `rating` via the mediators in the revised graph: `drink_level`, `color`, `Upayment`. `personality`, `age_group`, `activity`, and `User_cuisine` no longer have graph paths to `rating` and are excluded.

**Key results** (n = 1,161):

| Treatment | Total Mediated | Dominant Pathway | Interpretation |
|---|---|---|---|
| `drink_level` | via food + service | Food & Service | Drinking habits route entirely through quality mediators |
| `color` | partial | Food (direct also) | Direct effect on rating plus food-mediated path |
| `Upayment` | via food + service | Food & Service | Payment method effects fully absorbed by quality mediators |

### Phase 4: Heterogeneous Treatment Effects (Hijos)

`HijosHTEAnalyzer` estimates the causal effect of having children on rating. `hijos` does not appear in the revised causal graph (its association collapsed from 0.703 to 0.133 on the corrected dataset, below the 0.20 threshold), so the analysis is treated as a data-driven question rather than a graph-identified effect. Pathways are derived from the graph's mediator structure — **8 pathways across 4 categories** (`Direct`, `Via_Food`, `Via_Service`, `Via_Color`). Methods:

- Unadjusted ATE (simple difference in means)
- Confounder-adjusted ATE (5 graph-present confounders: `food_rating`, `service_rating`, `color`, `drink_level`, `Upayment`; bootstrap 95% CI)
- Inverse Probability Weighting (IPW) with stabilized weights
- Subgroup-specific effect estimation by Business Hours

> Results from this phase will update once the analysis is re-run on the corrected 1,161-row dataset. The previous estimates (IPW ATE +0.720, unadjusted +0.723) were produced on the 31,559-row fan-out and should be treated as preliminary.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/Causal_inference-Case_study.git
cd Causal_inference-Case_study

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Phase 1 — Build data pipeline (creates final_data.pkl)
python 01_Data_Analysis/data_pipeline.py

# Phase 2 — Build causal graph
python 02_Causal_Graph/causal_graph_module_complete.py

# Phase 2 — Run EDA correlation extraction
python 02_Causal_Graph/correlation_analyis.py

# Phase 3 — Run Shapley attribution analysis
python 03_Causal_ATTRIBUTIONAnalysis/Attribution_analysis_module.py

# Phase 3 — Run DoWhy formal causal attribution
python 03_Causal_ATTRIBUTIONAnalysis/causal_graph_attribution_module.py

# Phase 4 — Run mediation analysis
python 04_Mediation_HTE/mediation_analysis.py

# Phase 4 — Run hijos HTE analysis
python 04_Mediation_HTE/HTE_analysis.py
```

All scripts import data exclusively from `01_Data_Analysis/data_pipeline.py` via `get_initial_data()`. All outputs save to the script's own phase subfolder (`Reports/` or `results/`).

---

## Key Results & Insights

- **Dataset corrected**: collapsing the one-to-many join fan-out reduced the dataset from 31,559 → 1,161 rows (one per rating). The previous analyses were fit on 27.2x-duplicated observations, understating standard errors by ~√27 and manufacturing demographic-rating associations
- **Causal graph revised**: 78-edge, 31-node, 53-cycle graph replaced by a strict 16-edge DAG with 3 direct treatments (`food_rating`, `service_rating`, `color`). Five previously claimed treatments (`hijos`, `personality`, `height`, `interest`, `activity`) did not clear the 0.20 association threshold on corrected data
- **Food and service quality** remain the two dominant causal drivers — their associations with `rating` (0.692, 0.680) are robust across both the old and corrected datasets
- **DoWhy graph-informed estimates** are substantially lower than naive correlations (food_rating: 0.549 vs. 0.864 on original data), confirming that confounding inflates naive associations
- **Mediation treatments updated**: `drink_level`, `color`, and `Upayment` are the exogenous variables with graph paths to `rating` via the mediators; prior treatments (`personality`, `age_group`, `activity`, `User_cuisine`) are excluded
- **HTE (hijos) results are preliminary**: the re-run on the corrected dataset is pending; prior IPW ATE (+0.720) was estimated on the fan-out data

---

## Data Source:
UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/232/restaurant+consumer+data

## Author

**Ella Ndala**
ndallaella@gmail.com

---

## License

MIT License. See LICENSE for details.
