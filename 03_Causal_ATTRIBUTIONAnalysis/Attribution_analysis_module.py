#%%
"""
ATTRIBUTION ANALYSIS MODULE
=========================
Answers: "How much does each treatment contribute to the overall rating?"

Treatments (Direct Factors):
- color, height, hijos, interest, personality, food_rating, service_rating

This module computes:
1. Direct effects: Treatment → Rating
2. Total effects: Treatment → Mediators → Rating
3. Indirect effects: Total - Direct
4. Attribution %: Each treatment's contribution to R²
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import importlib.util
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import networkx as nx

# Ensure Matplotlib cache is writable in this project.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MPL_CACHE = _REPO_ROOT / ".mplconfig"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# Ensure project root, 01_Data_Analysis/, and 02_Causal_Graph/ are on sys.path.
_HERE        = Path(__file__).resolve().parent          # 03_Causal_ATTRIBUTIONAnalysis/
_RESULTS_DIR = _HERE / "Reports"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_PIPELINE_DIR = _REPO_ROOT / "01_Data_Analysis"
_GRAPH_DIR    = _REPO_ROOT / "02_Causal_Graph"
for _p in (_REPO_ROOT, _PIPELINE_DIR, _GRAPH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_pipeline import get_initial_data                         # Phase 1 — canonical data source
from causal_graph_module_complete import (                         # Phase 2 — causal graph
    define_causal_graph_custom,
    correlation_edges,
)

#%%
df = get_initial_data()
graph, treatments = define_causal_graph_custom(correlation_edges, outcome='rating', visualize=False)
outcome = 'rating'
class MediationAnalyzer:
    """
    Analyzes how much each treatment explains the outcome through mediation analysis.
    """
    
    def __init__(self, df, treatments, outcome, graph=None):
        """
        Initialize the mediation analyzer.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with all variables
        treatments : list
            List of treatment variable names (direct factors to outcome)
        outcome : str
            Outcome variable name (e.g., 'rating')
        graph : nx.DiGraph, optional
            Causal graph for identifying mediators
        """
        self.df = df.copy()
        self.treatments = treatments
        self.outcome = outcome
        self.graph = graph
        
        # Results storage
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}  # Store encoders for categorical variables
        
        print("="*80)
        print("MEDIATION ANALYSIS INITIALIZED")
        print("="*80)
        print(f"Outcome: {outcome}")
        print(f"Treatments ({len(treatments)}): {treatments}")
        print(f"Sample size: {len(df):,}")
        
    
    def clean_data(self):
        """Clean and prepare data for analysis."""
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)
        
        # Get all relevant variables
        all_vars = self.treatments + [self.outcome]
        
        # Check which variables exist
        missing_vars = [v for v in all_vars if v not in self.df.columns]
        if missing_vars:
            print(f"⚠️  Warning: Missing variables: {missing_vars}")
            # Remove missing from treatments
            self.treatments = [t for t in self.treatments if t in self.df.columns]
            print(f"✓ Using available treatments: {self.treatments}")
        
        # Select relevant columns
        self.df = self.df[self.treatments + [self.outcome]].copy()
        
        # Encode categorical variables
        self.df_encoded = self.df.copy()
        for col in self.treatments:
            if self.df_encoded[col].dtype == 'object' or self.df_encoded[col].dtype.name == 'category':
                # Encode categorical variables
                le = LabelEncoder()
                self.df_encoded[col] = le.fit_transform(self.df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"✓ Encoded categorical variable: {col} ({len(le.classes_)} categories)")
        
        # Handle missing values
        initial_size = len(self.df_encoded)
        self.df_encoded = self.df_encoded.dropna()
        final_size = len(self.df_encoded)
        
        if initial_size > final_size:
            print(f"✓ Removed {initial_size - final_size:,} rows with missing values")
        
        print(f"✓ Final sample size: {final_size:,}")
        print(f"✓ Variables: {len(self.treatments)} treatments + 1 outcome")
        
        return self
    
    
    def compute_direct_effects(self):
        """
        Compute direct effect of each treatment on outcome.
        Direct Effect = coefficient in regression: outcome ~ treatment + other_treatments
        """
        print("\n" + "="*80)
        print("COMPUTING DIRECT EFFECTS")
        print("="*80)
        
        # Prepare data (use encoded data)
        X = self.df_encoded[self.treatments].values
        y = self.df_encoded[self.outcome].values
        
        # Standardize for interpretability
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit regression: rating ~ all treatments
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Store direct effects (standardized coefficients)
        self.results['direct_effects'] = {}
        self.results['model'] = model
        self.results['r2_total'] = model.score(X_scaled, y)
        
        for i, treatment in enumerate(self.treatments):
            coef = model.coef_[i]
            self.results['direct_effects'][treatment] = coef
            print(f"  • {treatment:25s} → {self.outcome}: {coef:+.4f}")
        
        print(f"\n✓ Total R² (all treatments): {self.results['r2_total']:.4f}")
        print(f"  ({self.results['r2_total']*100:.2f}% of variance explained)")
        
        return self
    
    
    def compute_individual_contributions(self):
        """
        Compute how much each treatment contributes to R².
        Uses sequential R² decomposition.
        """
        print("\n" + "="*80)
        print("COMPUTING INDIVIDUAL CONTRIBUTIONS")
        print("="*80)
        
        X = self.df_encoded[self.treatments].values
        y = self.df_encoded[self.outcome].values
        X_scaled = self.scaler.transform(X)
        
        # Baseline: no treatments
        baseline_predictions = np.mean(y)
        baseline_ss = np.sum((y - baseline_predictions)**2)
        
        # Method 1: Unique R² contribution (partial R²)
        # For each treatment, compare R² with vs without it
        
        contributions = {}
        
        for i, treatment in enumerate(self.treatments):
            # Model without this treatment
            other_treatments = [t for t in self.treatments if t != treatment]
            other_indices = [j for j, t in enumerate(self.treatments) if t != treatment]
            
            if len(other_treatments) > 0:
                X_without = X_scaled[:, other_indices]
                model_without = LinearRegression()
                model_without.fit(X_without, y)
                r2_without = model_without.score(X_without, y)
            else:
                r2_without = 0
            
            # R² with all treatments (already computed)
            r2_with = self.results['r2_total']
            
            # Unique contribution
            unique_contribution = r2_with - r2_without
            contributions[treatment] = unique_contribution
            
            print(f"  • {treatment:25s}: {unique_contribution:.4f} ({unique_contribution/r2_with*100:.2f}% of total R²)")
        
        self.results['contributions'] = contributions
        
        # Compute percentages
        total_contribution = sum(contributions.values())
        percentages = {t: (c/total_contribution*100 if total_contribution > 0 else 0) 
                      for t, c in contributions.items()}
        self.results['contribution_percentages'] = percentages
        
        print(f"\n✓ Sum of unique contributions: {total_contribution:.4f}")
        
        return self
    
    
    def compute_shapley_values(self):
        """
        Compute Shapley values for fair attribution.
        More robust than sequential R² but computationally intensive.
        """
        print("\n" + "="*80)
        print("COMPUTING SHAPLEY VALUES (Fair Attribution)")
        print("="*80)
        
        from itertools import combinations
        
        X = self.df_encoded[self.treatments].values
        y = self.df_encoded[self.outcome].values
        X_scaled = self.scaler.transform(X)
        
        n_treatments = len(self.treatments)
        shapley_values = {t: 0.0 for t in self.treatments}
        
        # For each treatment
        for i, treatment in enumerate(self.treatments):
            marginal_contributions = []
            
            # For each possible subset not containing this treatment
            other_treatments = [t for j, t in enumerate(self.treatments) if j != i]
            
            for r in range(len(other_treatments) + 1):
                for subset in combinations(range(len(other_treatments)), r):
                    # Subset without treatment
                    subset_indices = [self.treatments.index(other_treatments[j]) for j in subset]
                    
                    # R² without treatment
                    if len(subset_indices) > 0:
                        X_subset = X_scaled[:, subset_indices]
                        model_subset = LinearRegression()
                        model_subset.fit(X_subset, y)
                        r2_without = model_subset.score(X_subset, y)
                    else:
                        r2_without = 0
                    
                    # R² with treatment added
                    subset_with_treatment = subset_indices + [i]
                    X_with = X_scaled[:, subset_with_treatment]
                    model_with = LinearRegression()
                    model_with.fit(X_with, y)
                    r2_with = model_with.score(X_with, y)
                    
                    # Marginal contribution
                    marginal = r2_with - r2_without
                    
                    # Weight by subset size
                    weight = 1.0 / (n_treatments * len(list(combinations(range(n_treatments-1), r))))
                    marginal_contributions.append(marginal * weight)
            
            shapley_values[treatment] = np.mean(marginal_contributions) * n_treatments
            print(f"  • {treatment:25s}: {shapley_values[treatment]:.4f}")
        
        self.results['shapley_values'] = shapley_values
        
        # Compute percentages
        total_shapley = sum(shapley_values.values())
        shapley_percentages = {t: (v/total_shapley*100 if total_shapley > 0 else 0) 
                              for t, v in shapley_values.items()}
        self.results['shapley_percentages'] = shapley_percentages
        
        print(f"\n✓ Sum of Shapley values: {total_shapley:.4f}")
        
        return self
    
    
    def compute_relative_importance(self):
        """
        Compute relative importance using dominance analysis.
        Simpler alternative to Shapley values.
        """
        print("\n" + "="*80)
        print("COMPUTING RELATIVE IMPORTANCE")
        print("="*80)
        
        # Use standardized coefficients weighted by correlation
        X = self.df_encoded[self.treatments].values
        y = self.df_encoded[self.outcome].values
        
        # Correlation of each treatment with outcome
        correlations = {}
        for treatment in self.treatments:
            corr = np.corrcoef(self.df_encoded[treatment], self.df_encoded[self.outcome])[0, 1]
            correlations[treatment] = corr
        
        # Relative importance = |coef| * |correlation|
        importance = {}
        for treatment in self.treatments:
            coef = abs(self.results['direct_effects'][treatment])
            corr = abs(correlations[treatment])
            importance[treatment] = coef * corr
        
        self.results['importance'] = importance
        
        # Normalize to percentages
        total_importance = sum(importance.values())
        importance_pct = {t: (imp/total_importance*100 if total_importance > 0 else 0) 
                         for t, imp in importance.items()}
        self.results['importance_percentages'] = importance_pct
        
        for treatment in self.treatments:
            print(f"  • {treatment:25s}: {importance[treatment]:.4f} ({importance_pct[treatment]:.2f}%)")
        
        return self
    
    
    def run_full_analysis(self):
        """Run complete mediation analysis pipeline."""
        print("\n" + "="*80)
        print("RUNNING FULL MEDIATION ANALYSIS")
        print("="*80)
        
        self.clean_data()
        self.compute_direct_effects()
        self.compute_individual_contributions()
        self.compute_relative_importance()
        
        # Only run Shapley if not too many treatments (computationally expensive)
        if len(self.treatments) <= 10:
            self.compute_shapley_values()
        else:
            print("\n⚠️  Skipping Shapley values (too many treatments)")
        
        self.summarize_results()
        
        return self
    
    
    def summarize_results(self):
        """Print comprehensive summary of results."""
        print("\n" + "="*80)
        print("SUMMARY: TREATMENT ATTRIBUTION TO RATING")
        print("="*80)
        
        print(f"\n📊 Overall Model Performance:")
        print(f"  • Total R²: {self.results['r2_total']:.4f}")
        print(f"  • Variance explained: {self.results['r2_total']*100:.2f}%")
        print(f"  • Treatments: {len(self.treatments)}")
        
        print(f"\n📈 Treatment Contributions (Relative Importance):")
        # Sort by importance
        sorted_treatments = sorted(self.results['importance_percentages'].items(), 
                                  key=lambda x: x[1], reverse=True)
        
        for i, (treatment, pct) in enumerate(sorted_treatments, 1):
            direct_effect = self.results['direct_effects'][treatment]
            contrib = self.results['contributions'][treatment]
            bar = "█" * int(pct / 2)  # Scale for display
            print(f"  {i}. {treatment:20s}: {pct:5.2f}% {bar}")
            print(f"     Direct effect: {direct_effect:+.4f}, Unique R²: {contrib:.4f}")
        
        if 'shapley_values' in self.results:
            print(f"\n🎯 Shapley Value Attribution:")
            sorted_shapley = sorted(self.results['shapley_percentages'].items(),
                                   key=lambda x: x[1], reverse=True)
            for treatment, pct in sorted_shapley:
                bar = "█" * int(pct / 2)
                print(f"  • {treatment:20s}: {pct:5.2f}% {bar}")
        
        print("\n" + "="*80)
    
    
    def create_attribution_chart(self, save_path='treatment_attribution.png'):
        """Create visualization of treatment attributions."""
        print(f"\n📊 Creating attribution visualization...")
        
        # Prepare data
        treatments = list(self.results['importance_percentages'].keys())
        percentages = list(self.results['importance_percentages'].values())
        direct_effects = [self.results['direct_effects'][t] for t in treatments]
        
        # Sort by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        treatments = [treatments[i] for i in sorted_indices]
        percentages = [percentages[i] for i in sorted_indices]
        direct_effects = [direct_effects[i] for i in sorted_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Percentage contribution
        colors = ['#FF6B6B' if i == 0 else '#4ECDC4' if i == 1 else '#95E1D3' 
                 for i in range(len(treatments))]
        
        bars1 = ax1.barh(treatments, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_xlabel('Contribution to Rating (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Treatment Attribution (Relative Importance)', fontsize=16, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add percentage labels
        for i, (treatment, pct) in enumerate(zip(treatments, percentages)):
            ax1.text(pct + 1, i, f'{pct:.1f}%', va='center', fontsize=11, fontweight='bold')
        
        # Plot 2: Direct effects
        effect_colors = ['#FF6B6B' if e > 0 else '#6B8EFF' for e in direct_effects]
        bars2 = ax2.barh(treatments, direct_effects, color=effect_colors, alpha=0.8, 
                        edgecolor='black', linewidth=2)
        ax2.set_xlabel('Direct Effect (Standardized)', fontsize=14, fontweight='bold')
        ax2.set_title('Direct Effects on Rating', fontsize=16, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        # Add effect labels
        for i, (treatment, effect) in enumerate(zip(treatments, direct_effects)):
            x_pos = effect + (0.02 if effect > 0 else -0.02)
            ha = 'left' if effect > 0 else 'right'
            ax2.text(x_pos, i, f'{effect:+.3f}', va='center', ha=ha, fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Attribution chart saved to '{save_path}'")
        
        return fig
    
    
    def create_summary_table(self):
        """Create summary table of all metrics."""
        print(f"\n📋 Creating summary table...")
        
        # Build summary dataframe
        summary_data = []
        for treatment in self.treatments:
            row = {
                'Treatment': treatment,
                'Direct Effect': self.results['direct_effects'][treatment],
                'Unique R²': self.results['contributions'][treatment],
                'Importance (%)': self.results['importance_percentages'][treatment],
            }
            
            if 'shapley_values' in self.results:
                row['Shapley (%)'] = self.results['shapley_percentages'][treatment]
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by importance
        summary_df = summary_df.sort_values('Importance (%)', ascending=False)
        
        print("\n" + "="*80)
        print("TREATMENT ATTRIBUTION SUMMARY TABLE")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        return summary_df


def run_mediation_analysis(df, treatments, outcome='rating', graph=None, 
                          save_chart=True, chart_path=None):
    """
    Convenience function to run full mediation analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with all variables
    treatments : list
        List of treatment variable names
    outcome : str
        Outcome variable name
    graph : nx.DiGraph, optional
        Causal graph
    save_chart : bool
        Whether to save attribution chart
    chart_path : str
        Path to save chart
    
    Returns:
    --------
    analyzer : MediationAnalyzer
        Fitted analyzer object
    summary_df : pd.DataFrame
        Summary table
    """
    # Initialize analyzer
    analyzer = MediationAnalyzer(df, treatments, outcome, graph)

    # Run full analysis
    analyzer.run_full_analysis()

    # Create visualizations
    if save_chart:
        if chart_path is None:
            chart_path = str(_RESULTS_DIR / "treatment_attribution.png")
        analyzer.create_attribution_chart(save_path=chart_path)
    
    # Get summary table
    summary_df = analyzer.create_summary_table()
    
    return analyzer, summary_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    analyzer, summary_df = run_mediation_analysis(
        df=df,
        treatments=treatments,
        outcome='rating',
        save_chart=True,
        chart_path=str(_RESULTS_DIR / "example_attribution.png"),
    )
    
    print("\n✅ Example analysis complete!")

# %%
