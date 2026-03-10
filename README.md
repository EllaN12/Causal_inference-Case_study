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
4. **How does graph-informed causal attribution differ from naive correlation-based attribution?** (DoWhy counterfactual estimation)

---

## Technical Skills Demonstrated

| Skill Area | Tools & Methods |
|---|---|
| **Causal Inference** | DoWhy, DAG construction, backdoor criterion, propensity score weighting, mediation analysis |
| **Python** | pandas, numpy, scikit-learn, networkx, matplotlib, seaborn, scipy, statsmodels |
| **Statistical Methods** | Linear regression, logistic regression, bootstrap confidence intervals, Shapley value decomposition, IPW estimation, F-tests |
| **Data Engineering** | Multi-source data merging (9 CSV files), feature engineering (geodesic distance, KMeans clustering, cuisine matching), categorical encoding, missing value imputation |
| **SQL-Adjacent Skills** | Complex joins across relational tables (users, restaurants, ratings, cuisines, payments, parking, hours) |
| **Visualization** | Causal graph visualization, mediation path diagrams, effect decomposition charts, attribution comparison plots |
| **EDA & Profiling** | Automated profiling with ydata-profiling, web scraping of EDA reports for correlation extraction |

---

## Project Structure

```
Causal_inference-Case_study/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/
│   └── raw/                           # 9 source CSV files
│       ├── rating_final.csv           # User-restaurant ratings (target)
│       ├── userprofile.csv            # User demographics & preferences
│       ├── usercuisine.csv            # User cuisine preferences
│       ├── userpayment.csv            # User payment methods
│       ├── geoplaces2.csv             # Restaurant profiles & locations
│       ├── chefmozcuisine.csv         # Restaurant cuisine specialties
│       ├── chefmozparking.csv         # Restaurant parking data
│       ├── chefmozaccepts.csv         # Restaurant accepted payments
│       └── chefmozhours4.csv          # Restaurant operating hours
│
├── src/
│   ├── causal_analysis_module/        # Core analysis library
│   │   ├── __init__.py
│   │   ├── analysis.py                # Data loading, preprocessing, feature engineering
│   │   ├── causal_graph_module_complete.py  # DAG construction & visualization
│   │   ├── causal_graph_attribution.py      # DoWhy-based causal attribution
│   │   ├── mediation_analysis_module.py     # Mediation & Shapley attribution
│   │   ├── HTE_analysis.py                  # Heterogeneous treatment effects
│   │   ├── correlation_analyis.py           # EDA correlation web scraping
│   │   ├── causal_model.py                  # MediationAnalysis pipeline
│   │   └── causal_model_2.py                # Extended causal model
│   │
│   ├── data_analysis/                 # Exploratory analysis scripts
│   │   └── analysis.py
│   │
│   ├── causal_analysis/               # DoWhy integration & graph specs
│   │   ├── causal_analysis.py         # Main orchestration script
│   │   ├── C_analysis.py              # DoWhy causal model setup
│   │   ├── causal_model_dowhy.py      # DoWhy model configuration
│   │   └── causal_graph_old.py        # Legacy graph definition
│   │
│   └── experiments/                   # Additional causal ML experiments
│       ├── 01_hotel_cancelations.py   # Hotel cancellation causal analysis
│       ├── 02_hotel_lead_time_experiment.py
│       └── causal_ml_experiments/     # Meta-learners, uplift trees, etc.
│
├── notebooks/
│   └── Interactive-1.ipynb            # Interactive exploration notebook
│
├── Reports/                           # Generated visualizations & outputs
│   ├── causal_graph.png               # Full causal DAG visualization
│   ├── attribution_analysis.png       # Treatment attribution chart
│   ├── food_rating_causal_effect.png  # Food rating causal effects
│   ├── treatment_effect_histogram.png # Treatment effect distribution
│   └── attribution_results.txt        # Exported attribution results
│
└── docs/                              # Additional documentation
```

---

## Data Pipeline

### 1. Data Ingestion & Merging

Nine relational CSV files are loaded and merged through a series of joins that mirror SQL-style operations:

- **User data**: `userprofile` LEFT JOIN `userpayment` LEFT JOIN `usercuisine` LEFT JOIN `rating_final`
- **Restaurant data**: `geoplaces2` LEFT JOIN `chefmozaccepts` LEFT JOIN `chefmozparking` LEFT JOIN `chefmozcuisine` LEFT JOIN `chefmozhours4`
- **Combined**: User data LEFT JOIN Restaurant data ON `placeID`

### 2. Feature Engineering

- **Patron-Restaurant Distance**: Geodesic distance (km) between user and restaurant coordinates using the Haversine formula
- **Location Clusters**: KMeans (k=5) clustering on restaurant latitude/longitude
- **Age Groups**: Derived from birth year, binned into 18-25, 26-35, 36-50, 50+
- **Cuisine Match Score**: Jaccard-style similarity (0-1) between user cuisine preferences and restaurant specialties
- **Business Hours Categories**: Parsed time strings categorized into Morning, Afternoon, Evening, Full Day, 24H

### 3. Preprocessing

- Categorical cleaning (replacing `?` markers with NaN, standardizing values)
- Missing value imputation (most-frequent strategy for categorical features)
- Label encoding for causal model compatibility

---

## Causal Analysis Framework

### Causal Graph (DAG)

A directed acyclic graph with **30+ nodes** and **90+ edges** defines the assumed causal structure. Edges were derived from domain knowledge combined with correlation analysis (via automated EDA profiling). The graph identifies **7 direct treatment variables** that causally affect the outcome (`rating`):

1. `food_rating` — Food quality score
2. `service_rating` — Service quality score
3. `personality` — User personality type
4. `color` — User color preference (proxy for personality traits)
5. `height` — User height
6. `hijos` — Whether user has children
7. `interest` — User interest category

### Attribution Methods

The project compares multiple attribution approaches:

- **Naive Correlation**: Simple Pearson correlation between treatment and outcome
- **Direct Effects (OLS)**: Standardized regression coefficients controlling for all treatments
- **Unique R² Contribution**: Partial R² (drop-one) decomposition
- **Shapley Values**: Game-theoretic fair attribution of R² across all treatment combinations
- **Relative Importance**: |coefficient| × |correlation| weighted attribution
- **DoWhy Causal Estimation**: Backdoor-adjusted linear regression via the DoWhy framework

### Mediation Analysis

For each treatment, effects are decomposed into:

- **Direct Effect** (c'): Treatment → Rating (controlling for mediators)
- **Indirect via Food** (a₁ × b₁): Treatment → Food Rating → Overall Rating
- **Indirect via Service** (a₂ × b₂): Treatment → Service Rating → Overall Rating
- **Total Mediated**: Sum of indirect effects as a proportion of total effect

### Heterogeneous Treatment Effects (HTE)

The `HijosHTEAnalyzer` module demonstrates HTE analysis for the `hijos` (children) variable across **43 identified causal pathways**, grouped into 7 interpretable categories. Methods include:

- Unadjusted ATE (simple difference in means)
- Confounder-adjusted ATE (with bootstrap 95% CI)
- Inverse Probability Weighting (IPW) with stabilized weights
- Subgroup-specific effect estimation

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/Causal_inference-Case_study.git
cd Causal_inference-Case_study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main causal analysis pipeline
cd src
python -m causal_analysis.causal_analysis

# Or run individual modules
python -m causal_analysis_module.causal_graph_module_complete  # Build & visualize DAG
python -m causal_analysis_module.mediation_analysis_module     # Run mediation analysis
```

---

## Key Results & Insights

- **Food quality and service quality** are the dominant mediators through which most treatment effects flow to the overall rating
- **Shapley value decomposition** reveals that `food_rating` and `service_rating` account for the majority of explained variance in ratings
- **Graph-informed causal estimates** (via DoWhy) differ meaningfully from naive correlations, highlighting the importance of controlling for confounders
- **Heterogeneous effects** show that the impact of having children (`hijos`) on ratings varies significantly across user subgroups, suggesting opportunities for targeted restaurant recommendations

---

## Relevance to Google BizOps Data Science Role

This project directly demonstrates the competencies outlined in the Google Ads & Commerce Finance Data Science Analyst position:

- **SQL and dashboard automation**: The multi-table data merging mirrors SQL JOIN operations; the modular pipeline structure supports automated report generation
- **Scalable financial tools and infrastructure**: The `causal_analysis_module` package is designed as reusable, extensible infrastructure — not one-off scripts
- **Data analysis with business insights**: Every statistical finding is paired with actionable interpretation (e.g., "invest in family-friendly features" based on HTE results)
- **Innovation in investigative frameworks**: Applying Shapley values and formal causal graphs to restaurant ratings demonstrates first-mover thinking on analytical methodology
- **Integration with data science community**: The modular architecture enables other analysts to import and extend the causal analysis toolkit

---

## Author

**Ella Ndala**
ndallaella@gmail.com

---

## License

This project is for portfolio and educational purposes.
