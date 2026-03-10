#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Canonical pathway map used by decomposition and reporting steps.
HIJOS_PATHWAYS = {
    "Direct": [
        ("hijos", "rating"),
    ],
    "Via_Food": [
        ("hijos", "food_rating", "rating"),
        ("hijos", "food_rating", "price", "rating"),
        ("hijos", "food_rating", "budget", "rating"),
        ("hijos", "food_rating", "service_rating", "rating"),
        ("hijos", "food_rating", "drink_level", "rating"),
        ("hijos", "food_rating", "ambience", "rating"),
        ("hijos", "food_rating", "dress_preference", "rating"),
        ("hijos", "food_rating", "smoker", "rating"),
    ],
    "Via_Service": [
        ("hijos", "service_rating", "rating"),
        ("hijos", "service_rating", "price", "rating"),
        ("hijos", "service_rating", "budget", "rating"),
        ("hijos", "service_rating", "food_rating", "rating"),
        ("hijos", "service_rating", "drink_level", "rating"),
        ("hijos", "service_rating", "ambience", "rating"),
        ("hijos", "service_rating", "dress_preference", "rating"),
        ("hijos", "service_rating", "smoker", "rating"),
    ],
    "Via_Color": [
        ("hijos", "color", "rating"),
        ("hijos", "color", "food_rating", "rating"),
        ("hijos", "color", "service_rating", "rating"),
        ("hijos", "color", "ambience", "rating"),
        ("hijos", "color", "drink_level", "rating"),
        ("hijos", "color", "dress_preference", "rating"),
        ("hijos", "color", "price", "rating"),
        ("hijos", "color", "budget", "rating"),
    ],
    "Via_Personality": [
        ("hijos", "personality", "rating"),
        ("hijos", "personality", "food_rating", "rating"),
        ("hijos", "personality", "service_rating", "rating"),
        ("hijos", "personality", "ambience", "rating"),
        ("hijos", "personality", "drink_level", "rating"),
        ("hijos", "personality", "smoker", "rating"),
    ],
    "Via_Height": [
        ("hijos", "height", "rating"),
        ("hijos", "height", "food_rating", "rating"),
        ("hijos", "height", "service_rating", "rating"),
        ("hijos", "height", "price", "rating"),
        ("hijos", "height", "budget", "rating"),
        ("hijos", "height", "ambience", "rating"),
    ],
    "Via_Interest": [
        ("hijos", "interest", "rating"),
        ("hijos", "interest", "food_rating", "rating"),
        ("hijos", "interest", "service_rating", "rating"),
        ("hijos", "interest", "drink_level", "rating"),
        ("hijos", "interest", "dress_preference", "rating"),
        ("hijos", "interest", "ambience", "rating"),
    ],
}



def _load_hijos_pathways_from_causal_graph(max_length=4):
    """
    Build hijos->rating pathways from the shared causal graph module.
    Returns None on any failure so static pathways remain the safe fallback.
    """
    try:
        import io
        import sys
        from contextlib import redirect_stdout

        project_root = Path(__file__).resolve().parents[1]
        graph_dir = project_root / "02_Causal_Graph"
        if str(graph_dir) not in sys.path:
            sys.path.insert(0, str(graph_dir))

        from causal_graph_module_complete import (
            correlation_edges,
            define_causal_graph_custom,
            get_causal_paths,
        )

        with redirect_stdout(io.StringIO()):
            graph, _ = define_causal_graph_custom(
                edges=correlation_edges,
                outcome="rating",
                visualize=False,
            )
        paths = get_causal_paths(graph, source="hijos", target="rating", max_length=max_length)
        if not paths:
            return None

        grouped = {
            "Direct": [],
            "Via_Food": [],
            "Via_Service": [],
            "Via_Color": [],
            "Via_Personality": [],
            "Via_Height": [],
            "Via_Interest": [],
        }

        for path in paths:
            path_tuple = tuple(path)
            if len(path_tuple) <= 2:
                grouped["Direct"].append(path_tuple)
                continue

            mediator_set = set(path_tuple[1:-1])
            if "food_rating" in mediator_set:
                grouped["Via_Food"].append(path_tuple)
            elif "service_rating" in mediator_set:
                grouped["Via_Service"].append(path_tuple)
            elif "color" in mediator_set:
                grouped["Via_Color"].append(path_tuple)
            elif "personality" in mediator_set:
                grouped["Via_Personality"].append(path_tuple)
            elif "height" in mediator_set:
                grouped["Via_Height"].append(path_tuple)
            elif "interest" in mediator_set:
                grouped["Via_Interest"].append(path_tuple)
            else:
                grouped["Direct"].append(path_tuple)

        for key in grouped:
            grouped[key] = sorted(set(grouped[key]))

        if sum(len(v) for v in grouped.values()) == 0:
            return None
        return grouped
    except Exception:
        return None


_graph_pathways = _load_hijos_pathways_from_causal_graph(max_length=4)
if _graph_pathways:
    HIJOS_PATHWAYS = _graph_pathways


class HijosHTEAnalyzer:
    """
    Comprehensive HTE analysis for hijos (having children) impact on rating.
    
    Handles 43 causal pathways with:
    1. Total effect estimation
    2. Path-specific decomposition
    3. Inverse probability weighting
    4. Heterogeneous effects by subgroups
    """
    
    def __init__(self, df, treatment='hijos', outcome='rating', confounders=None):
        """
        Initialize hijos HTE analyzer.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Restaurant data
        treatment : str
            Treatment variable (default: 'hijos')
        outcome : str
            Outcome variable (default: 'rating')
        confounders : list
            Variables to control for
        """
        self.df = df.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders or []
        
        print("="*80)
        print("HIJOS HTE ANALYSIS: 43 CAUSAL PATHWAYS")
        print("="*80)
        print(f"Treatment: {treatment} (having children)")
        print(f"Outcome: {outcome}")
        print(f"Confounders: {confounders}")
        print(f"\nCausal pathways:")
        for pathway_type, paths in HIJOS_PATHWAYS.items():
            print(f"  • {pathway_type}: {len(paths)} pathways")
        print(f"  • TOTAL: {sum(len(paths) for paths in HIJOS_PATHWAYS.values())} pathways")
    
    
    def estimate_total_effect(self):
        """
        Estimate total effect of hijos on rating (all pathways combined).
        
        Returns:
        --------
        total_effect_results : dict
            Total effect estimates
        """
        print("\n" + "="*80)
        print("STEP 1: TOTAL EFFECT (All 43 Pathways Combined)")
        print("="*80)
        
        # Prepare data
        cols_needed = [self.treatment, self.outcome] + self.confounders
        available_cols = [c for c in cols_needed if c in self.df.columns]
        
        df_clean = self.df[available_cols].dropna()
        print(f"\nSample size: {len(df_clean)}")
        
        # Ensure treatment is binary
        if df_clean[self.treatment].dtype == 'object':
            # Map to 0/1
            df_clean[self.treatment] = (df_clean[self.treatment].isin(['yes', 'dependent', 'dependents', 1])).astype(int)
        
        # Group sizes
        has_children = df_clean[df_clean[self.treatment] == 1]
        no_children = df_clean[df_clean[self.treatment] == 0]
        
        print(f"\nSample breakdown:")
        print(f"  • Has children: {len(has_children)} ({len(has_children)/len(df_clean)*100:.1f}%)")
        print(f"  • No children: {len(no_children)} ({len(no_children)/len(df_clean)*100:.1f}%)")
        
        # Simple comparison (unadjusted)
        mean_with = has_children[self.outcome].mean()
        mean_without = no_children[self.outcome].mean()
        unadjusted_ate = mean_with - mean_without
        
        # T-test
        t_stat, p_value = stats.ttest_ind(has_children[self.outcome], 
                                         no_children[self.outcome])
        
        print(f"\n📊 Unadjusted Comparison:")
        print(f"  • With children: {mean_with:.3f}")
        print(f"  • Without children: {mean_without:.3f}")
        print(f"  • Difference: {unadjusted_ate:+.3f}")
        print(f"  • T-statistic: {t_stat:.3f}")
        print(f"  • P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ Statistically significant")
        else:
            print(f"  ~ Not statistically significant")
        
        # Adjusted for confounders (if provided)
        if self.confounders:
            print(f"\n📊 Confounder-Adjusted Estimate:")
            
            # Encode confounders
            confounder_df = df_clean[self.confounders].copy()
            for col in self.confounders:
                if confounder_df[col].dtype == 'object':
                    confounder_df[col] = pd.Categorical(confounder_df[col]).codes
            
            # Regression: rating ~ hijos + confounders
            X = pd.concat([df_clean[[self.treatment]], confounder_df], axis=1)
            y = df_clean[self.outcome]
            
            model = LinearRegression()
            model.fit(X, y)
            
            adjusted_ate = model.coef_[0]
            r2 = model.score(X, y)
            
            print(f"  • Adjusted ATE: {adjusted_ate:+.3f}")
            print(f"  • R²: {r2:.4f}")
            print(f"  • Controlling for: {', '.join(self.confounders)}")
            
            # Bootstrap confidence interval
            n_bootstrap = 1000
            boot_ates = []
            
            np.random.seed(42)
            for _ in range(n_bootstrap):
                boot_idx = np.random.choice(len(df_clean), size=len(df_clean), replace=True)
                boot_X = X.iloc[boot_idx]
                boot_y = y.iloc[boot_idx]
                
                boot_model = LinearRegression()
                boot_model.fit(boot_X, boot_y)
                boot_ates.append(boot_model.coef_[0])
            
            ci_lower = np.percentile(boot_ates, 2.5)
            ci_upper = np.percentile(boot_ates, 97.5)
            
            print(f"  • 95% CI: [{ci_lower:+.3f}, {ci_upper:+.3f}]")
            
            self.adjusted_ate = adjusted_ate
            self.adjusted_ci = (ci_lower, ci_upper)
        else:
            adjusted_ate = unadjusted_ate
            self.adjusted_ate = adjusted_ate
            self.adjusted_ci = None
        
        self.total_effect_results = {
            'unadjusted_ate': unadjusted_ate,
            'adjusted_ate': adjusted_ate,
            'mean_with_children': mean_with,
            'mean_without_children': mean_without,
            't_statistic': t_stat,
            'p_value': p_value,
            'n_with': len(has_children),
            'n_without': len(no_children)
        }
        
        return self.total_effect_results
    
    
    def calculate_propensity_scores(self):
        """
        Calculate propensity scores for having children.
        
        P(hijos = 1 | confounders)
        
        Returns:
        --------
        df_with_ps : pd.DataFrame
            Data with propensity scores
        """
        print("\n" + "="*80)
        print("STEP 2: PROPENSITY SCORE CALCULATION")
        print("="*80)
        
        if not self.confounders:
            print("⚠ No confounders specified, skipping propensity scores")
            return None
        
        # Prepare data
        cols_needed = [self.treatment, self.outcome] + self.confounders
        available_cols = [c for c in cols_needed if c in self.df.columns]
        
        df_clean = self.df[available_cols].dropna()
        
        # Ensure treatment is binary
        if df_clean[self.treatment].dtype == 'object':
            df_clean[self.treatment] = (df_clean[self.treatment].isin(['yes', 'dependent', 'dependents', 1])).astype(int)
        
        # Encode confounders
        confounder_df = df_clean[self.confounders].copy()
        for col in self.confounders:
            if confounder_df[col].dtype == 'object':
                confounder_df[col] = pd.Categorical(confounder_df[col]).codes
        
        X = confounder_df.values
        y = df_clean[self.treatment].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Logistic regression for propensity scores
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X_scaled, y)
        
        # Get propensity scores
        propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
        
        df_clean['propensity_score'] = propensity_scores
        
        print(f"\n📊 Propensity Score Distribution:")
        print(f"  Mean: {propensity_scores.mean():.3f}")
        print(f"  Std: {propensity_scores.std():.3f}")
        print(f"  Range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")
        
        # Check overlap
        treated_ps = propensity_scores[y == 1]
        control_ps = propensity_scores[y == 0]
        
        print(f"\n📊 Common Support:")
        print(f"  Treated: [{treated_ps.min():.3f}, {treated_ps.max():.3f}]")
        print(f"  Control: [{control_ps.min():.3f}, {control_ps.max():.3f}]")
        
        overlap_min = max(treated_ps.min(), control_ps.min())
        overlap_max = min(treated_ps.max(), control_ps.max())
        
        if overlap_min < overlap_max:
            print(f"  ✓ Overlap: [{overlap_min:.3f}, {overlap_max:.3f}]")
            in_overlap = (propensity_scores >= overlap_min) & (propensity_scores <= overlap_max)
            print(f"  • {in_overlap.sum()} observations ({in_overlap.mean()*100:.1f}%) in overlap region")
        else:
            print(f"  ⚠ Limited overlap")
        
        self.df_with_ps = df_clean
        self.ps_model = ps_model
        self.scaler = scaler
        
        return df_clean
    
    
    def estimate_ipw_effect(self):
        """
        Estimate treatment effect using Inverse Probability Weighting.
        
        Returns:
        --------
        ipw_results : dict
            IPW-weighted treatment effect
        """
        if not hasattr(self, 'df_with_ps'):
            self.calculate_propensity_scores()
        
        if self.df_with_ps is None:
            print("⚠ Cannot estimate IPW without propensity scores")
            return None
        
        print("\n" + "="*80)
        print("STEP 3: INVERSE PROBABILITY WEIGHTING (IPW)")
        print("="*80)
        
        df = self.df_with_ps.copy()
        
        # Calculate IPW weights
        # Weight = 1/PS for treated, 1/(1-PS) for control
        df['ipw_weight'] = np.where(
            df[self.treatment] == 1,
            1.0 / df['propensity_score'],
            1.0 / (1.0 - df['propensity_score'])
        )
        
        # Stabilize weights (trim extreme values)
        df['ipw_weight'] = df['ipw_weight'].clip(
            lower=df['ipw_weight'].quantile(0.01),
            upper=df['ipw_weight'].quantile(0.99)
        )
        
        print(f"\n📊 IPW Weights:")
        print(df['ipw_weight'].describe())
        
        # Weighted outcomes
        treated_weighted = np.average(
            df[df[self.treatment] == 1][self.outcome],
            weights=df[df[self.treatment] == 1]['ipw_weight']
        )
        
        control_weighted = np.average(
            df[df[self.treatment] == 0][self.outcome],
            weights=df[df[self.treatment] == 0]['ipw_weight']
        )
        
        ipw_ate = treated_weighted - control_weighted
        
        print(f"\n📊 IPW-Weighted Results:")
        print(f"  • Weighted mean (with children): {treated_weighted:.3f}")
        print(f"  • Weighted mean (without children): {control_weighted:.3f}")
        print(f"  • IPW ATE: {ipw_ate:+.3f}")
        
        # Bootstrap SE
        n_bootstrap = 1000
        boot_ates = []
        
        np.random.seed(42)
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(len(df), size=len(df), replace=True)
            boot_df = df.iloc[boot_idx]
            
            boot_treated = np.average(
                boot_df[boot_df[self.treatment] == 1][self.outcome],
                weights=boot_df[boot_df[self.treatment] == 1]['ipw_weight']
            )
            
            boot_control = np.average(
                boot_df[boot_df[self.treatment] == 0][self.outcome],
                weights=boot_df[boot_df[self.treatment] == 0]['ipw_weight']
            )
            
            boot_ates.append(boot_treated - boot_control)
        
        se = np.std(boot_ates)
        ci_lower = np.percentile(boot_ates, 2.5)
        ci_upper = np.percentile(boot_ates, 97.5)
        
        # T-test
        t_stat = ipw_ate / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(df)-1))
        
        print(f"  • SE: {se:.4f}")
        print(f"  • 95% CI: [{ci_lower:+.3f}, {ci_upper:+.3f}]")
        print(f"  • T-statistic: {t_stat:.3f}")
        print(f"  • P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ Statistically significant")
        else:
            print(f"  ~ Not statistically significant")
        
        self.ipw_results = {
            'ipw_ate': ipw_ate,
            'se': se,
            'ci': (ci_lower, ci_upper),
            't_stat': t_stat,
            'p_value': p_value,
            'treated_mean': treated_weighted,
            'control_mean': control_weighted
        }
        
        return self.ipw_results
    
    
    def decompose_pathways(self):
        """
        Decompose total effect into pathway-specific contributions.
        
        Groups 43 pathways into interpretable categories:
        - Direct
        - Via Food Quality
        - Via Service Quality
        - Via Color/Preference
        - Via Personality
        - Via Height
        - Via Interest
        
        Returns:
        --------
        pathway_decomposition : pd.DataFrame
            Contribution of each pathway category
        """
        print("\n" + "="*80)
        print("STEP 4: PATHWAY DECOMPOSITION (43 Pathways → 7 Categories)")
        print("="*80)
        
        # Get all mediators from pathways
        all_mediators = set()
        for pathway_type, paths in HIJOS_PATHWAYS.items():
            for path in paths:
                # Mediators are all variables except first (hijos) and last (rating)
                mediators = path[1:-1]
                all_mediators.update(mediators)
        all_mediators = sorted(all_mediators)
        
        print(f"\nIdentified {len(all_mediators)} unique mediators:")
        print(f"  {all_mediators}")
        
        # Estimate pathway contributions
        # Use sequential mediation analysis
        
        cols_needed = [self.treatment, self.outcome] + list(all_mediators) + self.confounders
        # Preserve order but remove duplicates to avoid duplicate-column DataFrame slices.
        cols_needed = list(dict.fromkeys(cols_needed))
        available_cols = [c for c in cols_needed if c in self.df.columns]
        missing_mediators = set(all_mediators) - set(available_cols)
        
        if missing_mediators:
            print(f"\n⚠ Missing mediators in data: {missing_mediators}")
            print(f"  Using available mediators only")
            all_mediators = [m for m in all_mediators if m in available_cols]
        
        df_clean = self.df[available_cols].dropna()
        
        # Ensure treatment is binary
        if df_clean[self.treatment].dtype == 'object':
            df_clean[self.treatment] = (df_clean[self.treatment].isin(['yes', 'dependent', 'dependents', 1])).astype(int)
        
        # Encode categorical variables
        for col in list(dict.fromkeys(all_mediators + self.confounders)):
            if col in df_clean.columns:
                col_data = df_clean[col]
                if isinstance(col_data, pd.Series) and col_data.dtype == 'object':
                    df_clean[col] = pd.Categorical(col_data).codes
        
        print(f"\n🔍 Estimating pathway contributions...")
        
        pathway_results = []
        
        # For each pathway category, estimate its contribution
        for pathway_type, paths in HIJOS_PATHWAYS.items():
            print(f"\n  • {pathway_type}: {len(paths)} pathways")
            
            # Get mediators for this pathway type
            pathway_mediators = set()
            for path in paths:
                pathway_mediators.update(path[1:-1])
            
            pathway_mediators = [m for m in pathway_mediators if m in df_clean.columns]
            
            if not pathway_mediators:
                print(f"    ⚠ No mediators available")
                continue
            
            # Model 1: hijos → mediators
            path_a_effects = {}
            for mediator in pathway_mediators:
                try:
                    X_a = df_clean[[self.treatment]]
                    y_a = df_clean[mediator]
                    
                    model_a = LinearRegression()
                    model_a.fit(X_a, y_a)
                    
                    path_a_effects[mediator] = model_a.coef_[0]
                except:
                    path_a_effects[mediator] = 0.0
            
            # Model 2: mediators → rating
            try:
                X_b = df_clean[pathway_mediators]
                y_b = df_clean[self.outcome]
                
                model_b = LinearRegression()
                model_b.fit(X_b, y_b)
                
                # Indirect effect = sum of (path_a × path_b) for each mediator
                indirect_effect = 0.0
                for i, mediator in enumerate(pathway_mediators):
                    indirect_effect += path_a_effects.get(mediator, 0) * model_b.coef_[i]
                
                pathway_results.append({
                    'Pathway_Type': pathway_type,
                    'N_Paths': len(paths),
                    'N_Mediators': len(pathway_mediators),
                    'Indirect_Effect': indirect_effect,
                    'Mediators': ', '.join(pathway_mediators[:3]) + ('...' if len(pathway_mediators) > 3 else '')
                })
                
                print(f"    Indirect effect: {indirect_effect:+.4f}")
            except Exception as e:
                print(f"    ⚠ Could not estimate: {e}")
        
        pathway_df = pd.DataFrame(pathway_results)
        pathway_df = pathway_df.sort_values('Indirect_Effect', ascending=False, key=abs)
        
        # Add proportion
        total_indirect = pathway_df['Indirect_Effect'].sum()
        pathway_df['Proportion'] = pathway_df['Indirect_Effect'] / total_indirect
        
        print(f"\n📊 Pathway Decomposition Results:")
        print("="*80)
        print(pathway_df.to_string(index=False))
        print("="*80)
        
        print(f"\n💡 Key Insights:")
        if len(pathway_df) > 0:
            strongest = pathway_df.iloc[0]
            print(f"  • STRONGEST pathway: {strongest['Pathway_Type']}")
            print(f"    - {strongest['N_Paths']} pathways, {strongest['N_Mediators']} mediators")
            print(f"    - Contributes {strongest['Indirect_Effect']:+.4f} ({abs(strongest['Proportion'])*100:.1f}%)")
        
        self.pathway_decomposition = pathway_df
        
        return pathway_df
    
    
    def estimate_heterogeneous_effects(self, subgroup_var=None):
        """
        Estimate heterogeneous treatment effects by subgroups.
        
        Parameters:
        -----------
        subgroup_var : str
            Variable to define subgroups (e.g., 'age_group', 'area')
        
        Returns:
        --------
        het_effects : pd.DataFrame
            Treatment effects by subgroup
        """
        print("\n" + "="*80)
        print("STEP 5: HETEROGENEOUS TREATMENT EFFECTS")
        print("="*80)
        
        if subgroup_var is None:
            print("No subgroup variable specified")
            print("Estimating by treatment status only")
            return None
        
        if subgroup_var not in self.df.columns:
            print(f"⚠ {subgroup_var} not in data")
            return None
        
        print(f"\nEstimating HTE by: {subgroup_var}")
        
        # Prepare data
        cols_needed = [self.treatment, self.outcome, subgroup_var] + self.confounders
        available_cols = [c for c in cols_needed if c in self.df.columns]
        
        df_clean = self.df[available_cols].dropna()
        
        # Ensure treatment is binary
        if df_clean[self.treatment].dtype == 'object':
            df_clean[self.treatment] = (df_clean[self.treatment].isin(['yes', 'dependent', 'dependents', 1])).astype(int)
        
        # For each subgroup, estimate effect
        het_results = []
        
        for subgroup in df_clean[subgroup_var].unique():
            subgroup_df = df_clean[df_clean[subgroup_var] == subgroup]
            
            if len(subgroup_df) < 20:
                continue
            
            # Simple ATE for this subgroup
            treated = subgroup_df[subgroup_df[self.treatment] == 1][self.outcome]
            control = subgroup_df[subgroup_df[self.treatment] == 0][self.outcome]
            
            if len(treated) > 0 and len(control) > 0:
                ate = treated.mean() - control.mean()
                
                # T-test
                t_stat, p_val = stats.ttest_ind(treated, control)
                
                het_results.append({
                    'Subgroup': subgroup,
                    'N_Total': len(subgroup_df),
                    'N_Treated': len(treated),
                    'N_Control': len(control),
                    'ATE': ate,
                    'P_value': p_val,
                    'Significant': p_val < 0.05
                })
        
        het_df = pd.DataFrame(het_results)
        het_df = het_df.sort_values('ATE', ascending=False)
        
        print(f"\n📊 Heterogeneous Effects by {subgroup_var}:")
        print("="*80)
        print(het_df.to_string(index=False))
        print("="*80)
        
        # Identify where effect is strongest
        if len(het_df) > 0:
            strongest = het_df.iloc[0]
            weakest = het_df.iloc[-1]
            
            print(f"\n💡 Heterogeneity Insights:")
            print(f"  • STRONGEST effect in: {strongest['Subgroup']} (ATE = {strongest['ATE']:+.3f})")
            print(f"  • WEAKEST effect in: {weakest['Subgroup']} (ATE = {weakest['ATE']:+.3f})")
            print(f"  • Range: {strongest['ATE'] - weakest['ATE']:.3f} rating points")
        
        self.het_effects = het_df
        
        return het_df
    
    
    def visualize_results(self, save_path='hijos_hte_analysis.png'):
        """
        Visualize HTE analysis results.
        """
        print(f"\n📊 Creating comprehensive visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ===================================================================
        # Plot 1: Total effect comparison
        # ===================================================================
        
        ax = axes[0, 0]
        
        if hasattr(self, 'total_effect_results'):
            results = self.total_effect_results
            
            methods = ['Unadjusted', 'Adjusted']
            effects = [results['unadjusted_ate'], results.get('adjusted_ate', results['unadjusted_ate'])]
            colors = ['#3498db', '#2ecc71']
            
            bars = ax.bar(methods, effects, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            
            # Add error bars if CI available
            if hasattr(self, 'adjusted_ci') and self.adjusted_ci:
                ci_lower, ci_upper = self.adjusted_ci
                ax.errorbar([1], [effects[1]], 
                          yerr=[[effects[1]-ci_lower], [ci_upper-effects[1]]],
                          fmt='none', color='black', capsize=10, linewidth=2)
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_ylabel('Average Treatment Effect', fontsize=12, fontweight='bold')
            ax.set_title('Total Effect: Having Children → Rating\n(All 43 Pathways Combined)', 
                        fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (method, effect) in enumerate(zip(methods, effects)):
                ax.text(i, effect + 0.02, f'{effect:+.3f}', 
                       ha='center', fontsize=12, fontweight='bold')
        
        # ===================================================================
        # Plot 2: IPW comparison (if available)
        # ===================================================================
        
        ax = axes[0, 1]
        
        if hasattr(self, 'ipw_results') and self.ipw_results:
            ipw = self.ipw_results
            
            methods = ['Without Children', 'With Children']
            means = [ipw['control_mean'], ipw['treated_mean']]
            colors = ['#e74c3c', '#2ecc71']
            
            bars = ax.bar(methods, means, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            
            ax.set_ylabel('Mean Rating (IPW-Weighted)', fontsize=12, fontweight='bold')
            ax.set_title(f'IPW-Weighted Means\n(ATE = {ipw["ipw_ate"]:+.3f}, p={ipw["p_value"]:.4f})', 
                        fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (method, mean) in enumerate(zip(methods, means)):
                ax.text(i, mean + 0.05, f'{mean:.3f}', 
                       ha='center', fontsize=12, fontweight='bold')
        
        # ===================================================================
        # Plot 3: Pathway decomposition (if available)
        # ===================================================================
        
        ax = axes[1, 0]
        
        if hasattr(self, 'pathway_decomposition'):
            pathway_df = self.pathway_decomposition
            
            colors = ['#3498db' if e > 0 else '#e74c3c' for e in pathway_df['Indirect_Effect']]
            
            ax.barh(pathway_df['Pathway_Type'], pathway_df['Indirect_Effect'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Indirect Effect Contribution', fontsize=12, fontweight='bold')
            ax.set_title('Pathway Decomposition\n(7 Categories from 43 Paths)', 
                        fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (pathway, effect) in enumerate(zip(pathway_df['Pathway_Type'], 
                                                      pathway_df['Indirect_Effect'])):
                x_pos = effect + (0.002 if effect > 0 else -0.002)
                ha = 'left' if effect > 0 else 'right'
                ax.text(x_pos, i, f'{effect:+.4f}', 
                       va='center', ha=ha, fontsize=10, fontweight='bold')
        
        # ===================================================================
        # Plot 4: Heterogeneous effects (if available)
        # ===================================================================
        
        ax = axes[1, 1]
        
        if hasattr(self, 'het_effects') and self.het_effects is not None:
            het_df = self.het_effects
            
            colors = ['#2ecc71' if sig else '#95a5a6' for sig in het_df['Significant']]
            
            ax.barh(het_df['Subgroup'], het_df['ATE'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Treatment Effect (ATE)', fontsize=12, fontweight='bold')
            ax.set_title('Heterogeneous Effects by Subgroup\n(Green = Significant)', 
                        fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        # ===================================================================
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Visualization saved to '{save_path}'")
        
        return fig
    
    
    def generate_summary_report(self):
        """
        Generate comprehensive summary report.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY: HIJOS IMPACT ON RATING")
        print("="*80)
        
        if hasattr(self, 'total_effect_results'):
            results = self.total_effect_results
            
            print(f"\n🎯 TOTAL EFFECT (All 43 Pathways):")
            print(f"  • Unadjusted ATE: {results['unadjusted_ate']:+.3f}")
            if 'adjusted_ate' in results and results['adjusted_ate'] != results['unadjusted_ate']:
                print(f"  • Adjusted ATE: {results['adjusted_ate']:+.3f}")
            print(f"  • P-value: {results['p_value']:.4f}")
            
            if results['p_value'] < 0.05:
                direction = "INCREASES" if results['adjusted_ate'] > 0 else "DECREASES"
                print(f"\n  ✓ Having children significantly {direction} rating by {abs(results['adjusted_ate']):.3f} points")
            else:
                print(f"\n  ~ No significant effect detected")
        
        if hasattr(self, 'ipw_results') and self.ipw_results:
            ipw = self.ipw_results
            
            print(f"\n📊 CAUSAL ESTIMATE (IPW):")
            print(f"  • IPW ATE: {ipw['ipw_ate']:+.3f}")
            print(f"  • 95% CI: [{ipw['ci'][0]:+.3f}, {ipw['ci'][1]:+.3f}]")
            print(f"  • P-value: {ipw['p_value']:.4f}")
            print(f"  • This is the CAUSAL effect (confounding removed)")
        
        if hasattr(self, 'pathway_decomposition'):
            pathway_df = self.pathway_decomposition
            
            print(f"\n🛤️  PATHWAY DECOMPOSITION:")
            print(f"  • 43 pathways grouped into {len(pathway_df)} categories")
            
            if len(pathway_df) > 0:
                strongest = pathway_df.iloc[0]
                print(f"\n  STRONGEST pathway category:")
                print(f"    • {strongest['Pathway_Type']}")
                print(f"    • Contribution: {strongest['Indirect_Effect']:+.4f}")
                print(f"    • {strongest['N_Paths']} pathways, {strongest['N_Mediators']} mediators")
                
                for idx, row in pathway_df.iterrows():
                    pct = abs(row['Proportion']) * 100
                    print(f"\n  {row['Pathway_Type']}:")
                    print(f"    • Effect: {row['Indirect_Effect']:+.4f} ({pct:.1f}%)")
                    print(f"    • {row['N_Paths']} pathways through: {row['Mediators']}")
        
        if hasattr(self, 'het_effects') and self.het_effects is not None:
            het_df = self.het_effects
            
            if len(het_df) > 0:
                print(f"\n👥 HETEROGENEOUS EFFECTS:")
                strongest = het_df.iloc[0]
                weakest = het_df.iloc[-1]
                
                print(f"  • Effect varies by subgroup!")
                print(f"  • STRONGEST in: {strongest['Subgroup']} (ATE = {strongest['ATE']:+.3f})")
                print(f"  • WEAKEST in: {weakest['Subgroup']} (ATE = {weakest['ATE']:+.3f})")
                print(f"  • Range: {strongest['ATE'] - weakest['ATE']:.3f} rating points")
        
        print(f"\n💡 BUSINESS IMPLICATIONS:")
        
        if hasattr(self, 'total_effect_results'):
            ate = self.total_effect_results.get('adjusted_ate', self.total_effect_results['unadjusted_ate'])
            
            if ate > 0:
                print(f"  • Families rate restaurants {ate:+.3f} points higher")
                print(f"  • Invest in family-friendly features:")
                if hasattr(self, 'pathway_decomposition'):
                    pathway_df = self.pathway_decomposition
                    food_row = pathway_df[pathway_df['Pathway_Type'] == 'Via_Food']
                    service_row = pathway_df[pathway_df['Pathway_Type'] == 'Via_Service']
                    
                    if len(food_row) > 0:
                        food_contrib = food_row.iloc[0]['Indirect_Effect']
                        print(f"    - Food quality: {abs(food_contrib):+.4f} effect (kids menu, portions)")
                    
                    if len(service_row) > 0:
                        service_contrib = service_row.iloc[0]['Indirect_Effect']
                        print(f"    - Service quality: {abs(service_contrib):+.4f} effect (attentive, fast)")
            elif ate < 0:
                print(f"  • Families rate restaurants {abs(ate):.3f} points lower")
                print(f"  • Families may have higher standards or different needs")
                print(f"  • Opportunity to better serve this segment")
            else:
                print(f"  • No significant difference between families and non-families")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING HIJOS HTE ANALYSIS")
    print("=" * 80)

    try:
        # Reuse centralized data preparation pipeline.
        from data_pipeline import run_pipeline

        df = run_pipeline(save_output=False, run_eda_report=False)
        print(f"\nLoaded dataset for HTE analysis: {df.shape}")

        confounder_candidates = [
            "food_rating", "service_rating", "price", "smoker",
            "drink_level", "dress_preference", "budget", "ambience"
        ]
        confounders = [c for c in confounder_candidates if c in df.columns]

        analyzer = HijosHTEAnalyzer(
            df=df,
            treatment="hijos",
            outcome="rating",
            confounders=confounders
        )

        analyzer.estimate_total_effect()
        analyzer.calculate_propensity_scores()
        analyzer.estimate_ipw_effect()
        analyzer.decompose_pathways()

        subgroup_var = "Business_hours" if "Business_hours" in df.columns else None
        if subgroup_var:
            analyzer.estimate_heterogeneous_effects(subgroup_var=subgroup_var)

        analyzer.visualize_results(save_path="HTE_analysis_summary.png")
        analyzer.generate_summary_report()

        print("\n✓ HTE analysis completed successfully.")

    except Exception as exc:
        print(f"\n✗ HTE analysis failed: {exc}")
        raise
