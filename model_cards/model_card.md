





# Model Card for Restaurant Rating Causal Attribution — DoWhy, Mediation &amp; HTE

## Model Details

### Overview
Applies causal inference to a restaurant recommendation dataset (1,161 rows x 52 features from 9 merged relational files — one row per user-restaurant rating) to identify what causally drives overall ratings and through which pathways. Constructs a 14-node / 16-edge causal graph (a strict DAG, re-derived on the corrected table), identifies treatments and backdoor confounders against a Shapley-value correlational baseline, then decomposes effects with Baron &amp; Kenny mediation analysis and IPW-based heterogeneous treatment effects across 17 pathways grouped into 7 categories. Attribution now resolves to two treatments: food and service quality account for 99.5% of it. 

### Version

name: 1.2.0  

### Owners

* Ella Ndalla, ndallaella@gmail.com


### Licenses

* MIT

### References

* [https://archive.ics.uci.edu/dataset/232/restaurant+consumer+data](https://archive.ics.uci.edu/dataset/232/restaurant+consumer+data)


### Citations

* Ella Ndalla. Restaurant Rating Causal Attribution — DoWhy, Mediation &amp; HTE. GitHub repository: Causal_inference-Case_study.



## Considerations

### Users

* Data scientists working on attribution and experimentation

* Portfolio reviewers


### Use Cases

* Demonstrating how graph-informed causal attribution differs from naive correlation-based attribution.

* Methodological reference for mediation analysis and heterogeneous treatment effects.


### Limitations

* RESOLVED in v1.2.0. The graph is now a **strict DAG — 0 cycles**. The 53 cycles in v1.0.0/v1.1.0 came from directing within-layer pairs (user attribute ↔ restaurant attribute), whose association is selection rather than causation. Those pairs are now omitted by design and edges run forward through a fixed layering. See `docs/edge_derivation.md`.

* The edge list was re-derived on the corrected table. The previous 78 edges were screened by ydata-profiling on the 27.2x-inflated frame, which manufactured association between user demographics and rating: 42 of 75 evaluable edges lose their justification, and five of the seven original direct treatments collapse (personality 0.635 → 0.017, hijos 0.703 → 0.133, color 0.763 → 0.223, interest 0.555 → 0.198, height → 0.051). Re-running ydata-profiling on the corrected table confirms this independently — its alerts now report `rating` as correlated with food_rating and service_rating only.

* Only the two mediator edges clear ydata's own 0.50 alert threshold. The remaining 14 edges are weak association (0.20–0.41) retained so the graph has structure to identify against, and should be read as scaffolding rather than as findings. `color → rating` survives at 0.223 but contributes 0.53% of attribution — it is on the edge of not earning its place.

* **The mediation and HTE modules analyse treatments the current graph does not support.** `mediation_analysis.py` is hardcoded to age_group, activity, personality and User_cuisine; `HTE_analysis.py` to hijos. Of these, personality, User_cuisine and hijos are not nodes in the 14-node graph at all, and age_group and activity connect only through patron_restaurant_distance, not through the food/service mediators those modules decompose. Their results below are retained for continuity but are no longer graph-identified.

* Causal structure is assumed from domain knowledge plus automated correlation profiling; unmeasured confounding is not ruled out.

* Observational data only — no randomization or natural experiment supports the backdoor adjustment.

* All estimates in v1.0.0 were computed on a 31,559-row table produced by an un-collapsed left-join fan-out — 27.2x duplication of 1,161 real ratings. That inflated significance (standard errors understated by ~sqrt(27)) AND biased point estimates, because users and restaurants were implicitly weighted by how many cuisines and payment methods they listed. v1.1.0 re-estimates everything on the corrected table; effect sizes are roughly halved and two of four mediation treatments (personality, User_cuisine) no longer reach significance.

* DoWhy was the earlier estimation approach and has been superseded; `causal_graph_attribution_module.py` is no longer part of the pipeline. The v1.0.0 DoWhy backdoor figure (food_rating 0.549 vs. 0.864 naive) is therefore withdrawn rather than carried forward. The README and the `dowhy` import in `mediation_analysis.py` still reference it and are stale.

* The hijos HTE analysis rests on **19 treated observations** (1.6% of the sample). On the inflated table this read as 162 treated against 31,397 control, and every estimate looked comfortably significant. On the corrected table the confounder-adjusted ATE is +0.149 with a 95% CI of [−0.050, +0.332] — it does not exclude zero. Only the unadjusted difference and the IPW estimate remain significant, the latter at p = 0.031 with a CI whose lower bound is +0.016. This subgroup result should be read as suggestive at best.

* With n = 1,161 across 138 users and 130 restaurants, ratings are clustered within both. Standard errors here are unclustered and therefore still optimistic, though far less so than in v1.0.0.

* Demographic treatments (height, color preference, personality) are proxies with no plausible direct causal mechanism on restaurant ratings; significant estimates for them should be read as evidence of residual confounding, not as real effects.


### Tradeoffs

* Restricting the graph to associations the corrected data supports cuts it from 78 edges to 16 and from 7 treatments to 3 — trading breadth for edges that survive their own screen.

* IPW-based HTE gives subgroup detail but inflates variance in small strata.


### Ethical Considerations

* Risk: Discriminatory or spurious conclusions if demographic &#39;treatments&#39; (hijos, height, personality, color) are read as actionable causal levers.
  * Mitigation Strategy: All demographic effects are reported alongside mediation decomposition showing the substantive effects flow through food and service quality; the cycle-contaminated graph is disclosed.

* Risk: Business decisions made on non-identified estimates.
  * Mitigation Strategy: Naive Shapley attribution is reported side by side with the DoWhy estimate so the confounding gap is visible.

## Graphics



## Metrics

|Name|Value|
-----|------
|Analysis table|1,161 rows (one per user-restaurant rating), 138 users, 130 restaurants|
|Causal graph|14 nodes, 16 edges, 3 direct treatments on rating, **0 cycles (strict DAG)**|
|Total R² (all treatments)|0.6015|
|Attribution — food_rating|54.7% relative importance, Shapley 52.5%, direct effect +0.360|
|Attribution — service_rating|44.8% relative importance, Shapley 46.8%, direct effect +0.304|
|Attribution — color|0.5% relative importance, Shapley 0.7%, direct effect +0.024|
|Food + service share of attribution|99.5%|
|Mediation — activity through food/service|87.6% mediated (p = 0.0001) — not graph-identified|
|Mediation — age_group|101.3% mediated (p = 0.0053) — not graph-identified|
|Mediation — personality / User_cuisine|not significant (p = 0.7647 / p = 0.1309)|
|IPW ATE — hijos|+0.361, 95% CI [+0.016, +0.658] — not graph-identified|
|Confounder-adjusted ATE — hijos|+0.149, 95% CI [−0.050, +0.332] — **not significant**|
|Treated group size — hijos|19 of 1,161 (1.6%); 1,142 control|
