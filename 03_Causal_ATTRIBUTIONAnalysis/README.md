# Phase 4: Causal Analysis & Attribution

## Objective
Estimate causal effects of treatment variables on restaurant ratings using multiple attribution methods. Compare naive correlations against graph-informed causal estimates.

## Key Scripts
- `causal_analysis.py` — Main orchestration script for full causal analysis pipeline
- `C_analysis.py` — DoWhy causal model setup and effect estimation
- `causal_graph_attribution.py` — Multi-method attribution comparison
- `correlation_analyis.py` — EDA correlation extraction via automated profiling web scraping
- `causal_model.py` — Baseline causal model implementation

## Attribution Methods Compared
| Method | Description |
|--------|-------------|
| Naive Correlation | Pearson correlation between treatment and outcome |
| Direct Effects (OLS) | Standardized regression coefficients |
| Unique R² | Partial R² (drop-one) decomposition |
| Shapley Values | Game-theoretic fair attribution across all treatment combinations |
| Relative Importance | \|coefficient\| × \|correlation\| weighted attribution |
| DoWhy Causal | Backdoor-adjusted linear regression via DoWhy |

## Key Results
- Graph-informed (DoWhy) estimates differ meaningfully from naive correlations
- `food_rating` and `service_rating` dominate Shapley value attribution
- Controlling for confounders reduces apparent effect of demographic variables

## Outputs
- `Reports/Visualizations/attribution_analysis.png` — Side-by-side comparison of attribution methods
- `Reports/Visualizations/attribution_results.txt` — Exported attribution values
