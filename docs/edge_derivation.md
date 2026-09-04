# How `correlation_edges` is derived

Re-derived 2026-09-03. This document is the contract the edge list in
`02_Causal_Graph/causal_graph_module_complete.py` implements.

## Why the list was rebuilt

The original 78 edges were screened by ydata-profiling's "highly overall correlated"
alerts. That screen ran on the **31,559-row** table produced by an un-collapsed
left-join fan-out — 27.2x duplication of 1,161 real ratings. Each user was replicated
once per cuisine × payment method they happened to list, so their constant demographic
attributes were re-counted against their ratings, manufacturing association that is not
in the data.

Measured on both tables, 42 of 75 evaluable edges lose their justification. The
treatment layer is where it matters:

| Edge into `rating` | Inflated | Corrected | Kept |
|---|---|---|---|
| food_rating | 0.692 | 0.692 | ✅ |
| service_rating | 0.680 | 0.680 | ✅ |
| color | 0.763 | 0.223 | ⚠️ (weak) |
| interest | 0.555 | 0.198 | ❌ |
| hijos | 0.703 | 0.133 | ❌ |
| height | — | 0.051 | ❌ |
| personality | 0.635 | 0.017 | ❌ |

Re-running ydata-profiling on the corrected table confirms this independently: its
alerts pane now reports `rating` as correlated with **food_rating and one other field**
(service_rating) and with nothing else.

## The three rules

**1. Bias-corrected association.** Bergsma-corrected Cramér's V for categorical pairs,
Spearman for numeric pairs, correlation ratio (η) for mixed. The correction is not
optional here: plain Cramér's V inflates with cardinality, and collapsing the
one-to-many tables into `;`-delimited sets *raised* the cardinality of `User_cuisine`
(35 levels) and `restaurant_specialty` (30). Without the correction those two columns
become spurious hubs — the uncorrected screen alerts on `personality ↔ User_cuisine`,
`religion ↔ User_cuisine`, `transport ↔ User_cuisine` and so on, none of which are
structure. Categoricals above **15 levels** are excluded outright, as are identifiers
(`userID`, `placeID`, `zip`, `address`, …), raw coordinates, and the redundant
encodings of age (`birth_year`, `age` — `age_group` is retained).

**2. Layering, forward edges only.**

| Layer | Members |
|---|---|
| L0 exogenous | user attributes **and** restaurant attributes |
| L1 dyadic | `patron_restaurant_distance`, `cuisine_match_score` |
| L2 mediators | `food_rating`, `service_rating` |
| L3 outcome | `rating` |

Within-layer edges are omitted **by design**. A user's payment method does not cause a
restaurant's opening hours; that association is *selection* — which patron visits which
restaurant — not causation. The previous list directed such pairs and produced 53
cycles, which the model card correctly flagged as weakening every identification claim.
Forward-only layering makes the graph a strict DAG by construction: **0 cycles**.

**3. Inclusion threshold 0.20.**

## Read the threshold honestly

Only the two mediator edges clear ydata's own 0.50 alert level. Everything else in the
list is weak association, retained so the graph has structure to identify against.
There is a real gap between 0.680 and 0.223, and the list should be read as *two solid
edges plus scaffolding*, not sixteen equal findings.

| Threshold | Edges |
|---|---|
| 0.50 | 2 |
| 0.30 | 6 |
| **0.20** | **16** |
| 0.15 | 36 |

## Result

16 edges, 14 nodes, **strict DAG (0 cycles)**, three direct causes of `rating`:
`food_rating` (0.692), `service_rating` (0.680), `color` (0.223).

## Reproducing it

The screen needs `ydata-profiling` and `beautifulsoup4` — both declared in
`requirements.txt`. On Python 3.13 ydata-profiling also needs `setuptools<81`, because
it imports `pkg_resources`, which 3.13 no longer ships.

```bash
pip install -r requirements.txt "setuptools<81"
python 02_Causal_Graph/correlation_analyis.py     # regenerates Reports/eda_report.html
```
