





# Model Card for Restaurant Rating Causal Attribution — DoWhy, Mediation &amp; HTE

## Model Details

### Overview
Applies causal inference to a restaurant recommendation dataset (31,559 rows x 52 features from 9 merged relational files) to identify what causally drives overall ratings and through which pathways. Constructs a 31-node / 78-edge causal graph, estimates effects via DoWhy backdoor identification against a Shapley-value correlational baseline, then decomposes effects with Baron &amp; Kenny mediation analysis and IPW-based heterogeneous treatment effects across 17 pathways. 

### Version

name: 1.0.0  

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

* The causal graph contains 53 detected cycles and is therefore NOT a strict DAG. Cycles arise from feature-engineering interactions and are acknowledged, not resolved, in the DoWhy identification step — this weakens every identification claim.

* Causal structure is assumed from domain knowledge plus automated correlation profiling; unmeasured confounding is not ruled out.

* Observational data only — no randomization or natural experiment supports the backdoor adjustment.

* Demographic treatments (height, color preference, personality) are proxies with no plausible direct causal mechanism on restaurant ratings; significant estimates for them should be read as evidence of residual confounding, not as real effects.


### Tradeoffs

* DoWhy backdoor estimates are substantially more conservative than naive correlations (food_rating: 0.549 vs. 0.864) — the graph-informed estimate trades apparent effect size for defensibility.

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
|Causal graph|31 nodes, 78 edges, 7 direct treatments on rating, 53 cycles detected|
|Shapley attribution — food + service quality|~84% of treatment attribution|
|DoWhy causal effect — food_rating|0.549 (vs. 0.864 naive correlation)|
|Mediation — activity effect through food/service quality|100.2% (fully mediated)|
|IPW ATE — hijos (has children)|+0.720|
|Strongest HTE — morning-opening restaurants x hijos|ATE +1.476|

