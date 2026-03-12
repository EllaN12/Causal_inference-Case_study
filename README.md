# Causal Inference Case Study: Restaurant Rating Attribution Analysis

## Project Overview

This project applies **causal inference** techniques to a restaurant recommendation dataset to answer a core business question: **What factors causally drive overall restaurant ratings, and through which pathways do they operate?**

Rather than relying on correlations alone, this analysis constructs a formal causal graph (DAG), identifies treatment variables, estimates causal effects using DoWhy and custom pipelines, and decomposes those effects through mediation and heterogeneous treatment effect (HTE) analysis.

### Business Context

This project mirrors the analytical challenges faced by data science teams at companies like Google Ads & Commerce Finance — where understanding *why* metrics move (not just *that* they move) is essential for building scalable decision-making infrastructure. The methods demonstrated here translate directly to questions like revenue attribution, user behavior modeling, and A/B test analysis in production environments.

---

## Key Analytical Questions

1. **Which factors directly cause changes in restaurant ratings?** (Causal graph construction and attribution)
2. **How much of each treatment's effect flows through food quality vs. service quality?** (Mediation analysis)
3. **Do causal effects differ across customer subgroups?** (Heterogeneous treatment effects)
4. **How does graph-informed causal attribution differ from naive correlation-based attribution?** (DoWhy counterfactual estimation vs. Shapley decomposition)

---

## Technical Skills Demonstrated

| Skill Area | Tools & Methods |
|---|---|
| **Causal Inference** | DoWhy, DAG construction, backdoor criterion, propensity score weighting, mediation analysis, IPW |
| **Python** | pandas, numpy, scikit-learn, networkx, matplotlib, seaborn, scipy, statsmodels |
| **Statistical Methods** | OLS, logistic regression, bootstrap confidence intervals, Shapley value decomposition, IPW estimation, F-tests, ATE estimation |
| **Data Engineering** | Multi-source data merging (9 CSV files), feature engineering (geodesic distance, KMeans clustering, cuisine matching), categorical encoding, missing value imputation |
| **SQL-Adjacent Skills** | Complex joins across relational tables (users, restaurants, ratings, cuisines, payments, parking, hours) |
| **Visualization** | Causal graph visualization, mediation path diagrams, effect decomposition charts, attribution comparison plots |
| **EDA & Profiling** | Automated profiling with ydata-profiling, web scraping of EDA reports for correlation extraction |

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
│   ├── final_data.pkl                     # Cached output (31,559 rows × 52 features)
│   └── Reports/
│       └── data_pipeline_results.txt      # Pipeline run logs
│
├── 02_Causal_Graph/                       # Phase 2: Causal graph & correlation EDA
│   ├── causal_graph_module_complete.py    # DAG construction, visualization, path analysis
│   ├── correlation_analyis.py             # ydata-profiling EDA + BeautifulSoup extraction
│   └── Reports/
│       ├── causal_graph.png               # Full 31-node DAG visualization
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
└── causal_case_study_interview_prep/      # Legacy exploratory work (archived)
    ├── data/raw/                          # 9 source CSV files
    └── Reports/                           # Earlier output artifacts
```

---

## Data Pipeline (Phase 1)

All downstream analysis imports data exclusively via `get_initial_data()` from `01_Data_Analysis/data_pipeline.py`.

### 1. Data Ingestion & Merging

Nine relational CSV files are loaded and merged through SQL-style operations:

- **User data**: `userprofile` LEFT JOIN `userpayment` LEFT JOIN `usercuisine` LEFT JOIN `rating_final`
- **Restaurant data**: `geoplaces2` LEFT JOIN `chefmozaccepts` LEFT JOIN `chefmozparking` LEFT JOIN `chefmozcuisine` LEFT JOIN `chefmozhours4`
- **Combined**: User data LEFT JOIN Restaurant data ON `placeID`
- **Output**: 31,559 rows × 52 features

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

A directed graph with **31 nodes** and **78 edges** defines the assumed causal structure, derived from domain knowledge combined with correlation analysis via automated EDA profiling. The graph identifies **7 direct treatment variables** that causally affect `rating`:

1. `food_rating` — Food quality score
2. `service_rating` — Service quality score
3. `hijos` — Whether user has children
4. `height` — User height
5. `interest` — User interest category
6. `color` — User color preference (proxy for personality traits)
7. `personality` — User personality type

> **Note**: The graph contains 53 detected cycles (it is not a strict DAG). These arise from feature engineering interactions and are acknowledged in the DoWhy identification step via `proceed_when_unidentifiable=True`.

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

**Key results** (n = 31,559):

| Treatment | Total Mediated | Dominant Pathway | Interpretation |
|---|---|---|---|
| `activity` | 100.2% | Food (52.9%) | Fully mediated; negligible direct effect |
| `personality` | 72.6% | Food (75.2%) | Food-dominant pathway |
| `age_group` | 62.3% | Service (61.8%) | Service-dominant pathway |
| `User_cuisine` | 40.9% | Mostly direct | Cuisine preference acts directly |

### Phase 4: Heterogeneous Treatment Effects (Hijos)

`HijosHTEAnalyzer` analyses the causal effect of having children across **17 identified causal pathways** (dynamically extracted from the causal graph), grouped into 6 categories. Methods:

- Unadjusted ATE (simple difference in means)
- Confounder-adjusted ATE (8 confounders, bootstrap 95% CI)
- Inverse Probability Weighting (IPW) with stabilized weights
- Subgroup-specific effect estimation by Business Hours

**Key results**:

| Metric | Value |
|---|---|
| Unadjusted ATE | +0.723 (p < 0.0001) |
| Adjusted ATE | +0.206 (95% CI: [+0.138, +0.268]) |
| IPW ATE | +0.720 (95% CI: [+0.627, +0.814]) |
| Strongest subgroup | Morning (ATE = +1.476) |
| Weakest subgroup | Full Day (ATE = +0.430) |
| ATE Range | 1.047 rating points |

> Families with children rate restaurants **significantly HIGHER** — the direct partial effect is negative (−0.095) because mediators (food/service) absorb and reverse the sign. The total causal effect (IPW) is strongly positive.

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

- **Food and service quality** are the two dominant causal drivers, together explaining ~84% of treatment attribution (Shapley) and most mediated effects
- **DoWhy graph-informed estimates** are substantially lower than naive correlations (food_rating: 0.549 vs. 0.864), confirming that confounding inflates naive associations
- **Activity** is fully mediated (100.2%) through food and service quality — its effect on ratings operates entirely via these intermediate channels
- **Families with children** (hijos) rate restaurants significantly higher (+0.720 IPW ATE), despite a small negative partial direct effect when mediators are controlled for
- **Heterogeneous effects** show morning-opening restaurants see the strongest family effect (ATE = +1.476), suggesting targeted investment in morning family-friendly features

---

## Relevance to Data Science Roles

This project directly demonstrates:

- **Causal inference at scale**: Formal DAG construction, backdoor identification, and propensity-weighted estimation — not just correlation
- **Multi-table data engineering**: 9-source merging pipeline mirroring SQL JOIN patterns with feature engineering
- **Scalable modular design**: Each phase is an importable module with a canonical data interface (`get_initial_data()`) and deterministic output paths
- **Business-grounded insights**: Every statistical result is paired with actionable interpretation (e.g., "invest in family-friendly morning features" from HTE results)
- **Reproducibility**: Path-anchored outputs, cached pipeline data, and rerunnable scripts from any working directory

---

## Author

**Ella Ndala**
ndallaella@gmail.com

---

## License

This project is for portfolio and educational purposes.
