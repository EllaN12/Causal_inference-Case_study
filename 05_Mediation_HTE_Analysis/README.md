# Phase 5: Mediation Analysis & Heterogeneous Treatment Effects (HTE)

## Objective
Decompose treatment effects into direct and indirect pathways, and identify how effects vary across customer subgroups.

## Key Scripts
- `mediation_analysis_module.py` — Mediation analysis: decomposes each treatment's effect into direct and indirect paths through food_rating and service_rating
- `HTE_analysis.py` — Heterogeneous Treatment Effect analysis for `hijos` variable across 43 identified causal pathways
- `causal_model.py` — MediationAnalysis pipeline
- `causal_model_2.py` — Extended causal model with additional estimators

## Mediation Analysis
For each of the 7 treatment variables, effects decomposed into:
- **Direct Effect (c')**: Treatment → Rating (controlling for mediators)
- **Indirect via Food (a₁×b₁)**: Treatment → Food Rating → Rating
- **Indirect via Service (a₂×b₂)**: Treatment → Service Rating → Rating
- **Total Mediated %**: Proportion of total effect flowing through mediators

## HTE Analysis (`hijos` — having children)
- **Methods**: Unadjusted ATE, Confounder-adjusted ATE (bootstrap 95% CI), IPW with stabilized weights
- **43 causal pathways** across 7 interpretable categories
- Shows significant variation in rating impact across user subgroups

## Key Results
- Most treatment effects are predominantly mediated through food and service quality
- Direct demographic effects on ratings are small after mediation adjustment
- Customers with children (`hijos=y`) show meaningfully different rating patterns

## Outputs
- `Reports/Visualizations/food_rating_causal_effect.png`
- `Reports/Visualizations/treatment_effect_histogram.png`
- `Reports/Visualizations/treatment_effect_histogram_sorted.png`
