#%%
import networkx as nx
from dowhy import CausalModel
import pandas as pd
import os
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import re
import matplotlib.pyplot as plt
import networkx as nx
import pandas_flavor as pf
#import causal_analysis_module as cam
from causal_analysis_module.analysis import data_preprocessing

#%%


## Question 1: How much of the overall rating is explained by each of the treatments 

class MediationAnalysis:
    """
    Performs mediation analysis to decompose treatment effects into:
    1. Direct effect (Treatment → Outcome)
    2. Indirect effect via food_rating (Treatment → Food → Outcome)
    3. Indirect effect via service_rating (Treatment → Service → Outcome)
    """
    
    def __init__(self, df, outcome='rating', mediator_food='food_rating', 
                 mediator_service='service_rating'):
        """
        Initialize mediation analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with ratings data
        outcome : str
            Outcome variable (overall rating)
        mediator_food : str
            Food quality mediator
        mediator_service : str
            Service quality mediator
        """
        self.df = df.copy()
        self.outcome = outcome
        self.mediator_food = mediator_food
        self.mediator_service = mediator_service
        self.results = {}
        
    def analyze_treatment(self, treatment_col, treatment_name=None):
        """
        Perform mediation analysis for a single treatment variable
        
        Steps:
        1. Total Effect: Treatment → Outcome (without mediators)
        2. Mediation via Food: Treatment → Food → Outcome
        3. Mediation via Service: Treatment → Service → Outcome
        4. Direct Effect: Treatment → Outcome (controlling for both mediators)
        5. Calculate proportions mediated
        
        Parameters:
        -----------
        treatment_col : str
            Name of treatment variable
        treatment_name : str, optional
            Display name for treatment
        """
        
        if treatment_name is None:
            treatment_name = treatment_col
            
        print(f"\n{'='*80}")
        print(f"MEDIATION ANALYSIS: {treatment_name}")
        print(f"{'='*80}")
        
        # Prepare data
        df_analysis = self._prepare_data(treatment_col)
        
        if df_analysis is None:
            print(f"⚠ Skipping {treatment_name}: insufficient valid data")
            return None
        
        # Extract variables
        X_treatment = df_analysis[treatment_col].values.reshape(-1, 1)
        M_food = df_analysis[self.mediator_food].values
        M_service = df_analysis[self.mediator_service].values
        Y = df_analysis[self.outcome].values
        
        print(f"\nSample size: n = {len(df_analysis)}")
        
        # ====== STEP 1: Total Effect (c) ======
        # Y = c*X + e
        total_model = LinearRegression().fit(X_treatment, Y)
        c_total = total_model.coef_[0]
        
        # Get R² and p-value
        y_pred_total = total_model.predict(X_treatment)
        ss_res = np.sum((Y - y_pred_total) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2_total = 1 - (ss_res / ss_tot)
        
        # F-test for total effect
        n = len(Y)
        k = 1  # number of predictors
        f_stat = (r2_total / k) / ((1 - r2_total) / (n - k - 1))
        p_value_total = 1 - stats.f.cdf(f_stat, k, n - k - 1)
        
        print(f"\n--- TOTAL EFFECT (without mediators) ---")
        print(f"  Effect coefficient: {c_total:.4f}")
        print(f"  R²: {r2_total:.4f}")
        print(f"  p-value: {p_value_total:.4f}")
        
        # ====== STEP 2: Path a1 (Treatment → Food) ======
        # M_food = a1*X + e
        path_a1_model = LinearRegression().fit(X_treatment, M_food)
        a1 = path_a1_model.coef_[0]
        
        print(f"\n--- PATH: Treatment → Food Quality ---")
        print(f"  Coefficient (a1): {a1:.4f}")
        
        # ====== STEP 3: Path a2 (Treatment → Service) ======
        # M_service = a2*X + e
        path_a2_model = LinearRegression().fit(X_treatment, M_service)
        a2 = path_a2_model.coef_[0]
        
        print(f"\n--- PATH: Treatment → Service Quality ---")
        print(f"  Coefficient (a2): {a2:.4f}")
        
        # ====== STEP 4: Direct Effect (c') and b paths ======
        # Y = c'*X + b1*M_food + b2*M_service + e
        X_mediated = np.column_stack([X_treatment.flatten(), M_food, M_service])
        mediated_model = LinearRegression().fit(X_mediated, Y)
        
        c_direct = mediated_model.coef_[0]  # Direct effect
        b1 = mediated_model.coef_[1]  # Food → Rating
        b2 = mediated_model.coef_[2]  # Service → Rating
        
        # R² for mediated model
        y_pred_mediated = mediated_model.predict(X_mediated)
        ss_res_med = np.sum((Y - y_pred_mediated) ** 2)
        r2_mediated = 1 - (ss_res_med / ss_tot)
        
        print(f"\n--- CONTROLLING FOR MEDIATORS ---")
        print(f"  Direct effect (c'): {c_direct:.4f}")
        print(f"  Food → Rating (b1): {b1:.4f}")
        print(f"  Service → Rating (b2): {b2:.4f}")
        print(f"  R² (with mediators): {r2_mediated:.4f}")
        
        # ====== STEP 5: Indirect Effects ======
        indirect_food = a1 * b1
        indirect_service = a2 * b2
        total_indirect = indirect_food + indirect_service
        
        print(f"\n--- INDIRECT EFFECTS ---")
        print(f"  Via Food Quality: {indirect_food:.4f}")
        print(f"  Via Service Quality: {indirect_service:.4f}")
        print(f"  Total Indirect: {total_indirect:.4f}")
        
        # ====== STEP 6: Mediation Proportions ======
        # What % of total effect is mediated?
        if abs(c_total) > 0.001:
            prop_mediated_total = total_indirect / c_total
            prop_mediated_food = indirect_food / c_total
            prop_mediated_service = indirect_service / c_total
            prop_direct = c_direct / c_total
            
            print(f"\n--- PROPORTION OF EFFECT EXPLAINED ---")
            print(f"  By Food Quality: {prop_mediated_food:.1%} of total effect")
            print(f"  By Service Quality: {prop_mediated_service:.1%} of total effect")
            print(f"  Total Mediated: {prop_mediated_total:.1%} of total effect")
            print(f"  Direct (not via mediators): {prop_direct:.1%} of total effect")
            
            # Relative importance of mediators
            if abs(total_indirect) > 0.001:
                food_relative = indirect_food / total_indirect
                service_relative = indirect_service / total_indirect
                
                print(f"\n--- RELATIVE IMPORTANCE OF MEDIATORS ---")
                print(f"  Food Quality: {food_relative:.1%} of mediated effect")
                print(f"  Service Quality: {service_relative:.1%} of mediated effect")
        else:
            prop_mediated_total = 0
            prop_mediated_food = 0
            prop_mediated_service = 0
            prop_direct = 0
            food_relative = 0
            service_relative = 0
            
            print(f"\n⚠ Total effect is negligible - mediation percentages not meaningful")
        
        # ====== STEP 7: Statistical Tests ======
        # Sobel test for indirect effects
        # SE(indirect) ≈ sqrt(a²*SE(b)² + b²*SE(a)²)
        
        # For simplicity, using bootstrap confidence intervals would be better
        # Here we'll provide point estimates
        
        print(f"\n--- VARIANCE EXPLAINED ---")
        print(f"  Treatment alone: R² = {r2_total:.4f}")
        print(f"  With mediators: R² = {r2_mediated:.4f}")
        print(f"  Increase from mediators: ΔR² = {r2_mediated - r2_total:.4f}")
        
        # Store results
        self.results[treatment_name] = {
            'treatment': treatment_col,
            'n': len(df_analysis),
            'total_effect': c_total,
            'direct_effect': c_direct,
            'indirect_food': indirect_food,
            'indirect_service': indirect_service,
            'total_indirect': total_indirect,
            'path_a1': a1,
            'path_a2': a2,
            'path_b1': b1,
            'path_b2': b2,
            'prop_mediated_food': prop_mediated_food,
            'prop_mediated_service': prop_mediated_service,
            'prop_mediated_total': prop_mediated_total,
            'prop_direct': prop_direct,
            'food_relative': food_relative,
            'service_relative': service_relative,
            'r2_total': r2_total,
            'r2_mediated': r2_mediated,
            'p_value_total': p_value_total
        }
        
        return self.results[treatment_name]
    
    def _prepare_data(self, treatment_col):
        """
        Prepare data for analysis - handle categorical variables and missing values
        """
        # Check if treatment column exists
        if treatment_col not in self.df.columns:
            print(f"⚠ Column '{treatment_col}' not found in data")
            return None
        
        # Select relevant columns
        cols = [treatment_col, self.mediator_food, self.mediator_service, self.outcome]
        df_sub = self.df[cols].copy()
        
        # Drop missing values
        df_sub = df_sub.dropna()
        
        if len(df_sub) < 30:
            print(f"⚠ Insufficient data after removing missing values: n={len(df_sub)}")
            return None
        
        # Handle categorical variables - encode as numeric
        if df_sub[treatment_col].dtype == 'object' or df_sub[treatment_col].dtype.name == 'category':
            le = LabelEncoder()
            df_sub[treatment_col] = le.fit_transform(df_sub[treatment_col])
            print(f"  Encoded categorical variable: {treatment_col}")
            print(f"  Categories: {list(le.classes_)}")
        
        return df_sub
    
    def create_summary_table(self):
        """
        Create summary table of all mediation analyses
        """
        if not self.results:
            print("No results to summarize. Run analyze_treatment first.")
            return None
        
        summary_data = []
        for name, res in self.results.items():
            summary_data.append({
                'Treatment': name,
                'N': res['n'],
                'Total_Effect': res['total_effect'],
                'Direct_Effect': res['direct_effect'],
                'Indirect_Food': res['indirect_food'],
                'Indirect_Service': res['indirect_service'],
                'Pct_via_Food': res['prop_mediated_food'] * 100,
                'Pct_via_Service': res['prop_mediated_service'] * 100,
                'Pct_Mediated': res['prop_mediated_total'] * 100,
                'Food_Importance': res['food_relative'] * 100,
                'Service_Importance': res['service_relative'] * 100,
                'R2_Total': res['r2_total'],
                'R2_Mediated': res['r2_mediated']
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        print("\n" + "="*100)
        print("SUMMARY: MEDIATION ANALYSIS RESULTS")
        print("="*100)
        print("\nAll effects and percentages:")
        print(df_summary.to_string(index=False))
        
        return df_summary
    
    def plot_mediation_results(self, save_path='mediation_results.png'):
        """
        Visualize mediation analysis results
        """
        if not self.results:
            print("No results to plot")
            return None
        
        n_treatments = len(self.results)
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Prepare data
        treatment_names = list(self.results.keys())
        
        # Plot 1: Effect decomposition
        ax1 = fig.add_subplot(gs[0, :])
        
        direct_effects = [r['direct_effect'] for r in self.results.values()]
        indirect_food = [r['indirect_food'] for r in self.results.values()]
        indirect_service = [r['indirect_service'] for r in self.results.values()]
        
        x = np.arange(len(treatment_names))
        width = 0.6
        
        p1 = ax1.bar(x, direct_effects, width, label='Direct Effect', alpha=0.8)
        p2 = ax1.bar(x, indirect_food, width, bottom=direct_effects, 
                    label='Indirect via Food', alpha=0.8)
        p3 = ax1.bar(x, indirect_service, width,
                    bottom=np.array(direct_effects) + np.array(indirect_food),
                    label='Indirect via Service', alpha=0.8)
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_ylabel('Effect Size', fontsize=12)
        ax1.set_xlabel('Treatment Variable', fontsize=12)
        ax1.set_title('Effect Decomposition: Direct vs Mediated Effects', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(treatment_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Proportion mediated
        ax2 = fig.add_subplot(gs[1, 0])
        
        prop_food = [r['prop_mediated_food'] * 100 for r in self.results.values()]
        prop_service = [r['prop_mediated_service'] * 100 for r in self.results.values()]
        prop_direct = [r['prop_direct'] * 100 for r in self.results.values()]
        
        # Stacked bar chart
        p1 = ax2.barh(x, prop_direct, height=0.6, label='Direct', alpha=0.8)
        p2 = ax2.barh(x, prop_food, height=0.6, left=prop_direct, 
                     label='Via Food', alpha=0.8)
        p3 = ax2.barh(x, prop_service, height=0.6, 
                     left=np.array(prop_direct) + np.array(prop_food),
                     label='Via Service', alpha=0.8)
        
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Percentage of Total Effect (%)', fontsize=11)
        ax2.set_title('Proportion of Effect Explained', fontsize=12, fontweight='bold')
        ax2.set_yticks(x)
        ax2.set_yticklabels(treatment_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Relative importance of mediators
        ax3 = fig.add_subplot(gs[1, 1])
        
        food_importance = [r['food_relative'] * 100 for r in self.results.values()]
        service_importance = [r['service_relative'] * 100 for r in self.results.values()]
        
        x_pos = np.arange(len(treatment_names))
        width = 0.35
        
        ax3.bar(x_pos - width/2, food_importance, width, label='Food Quality', alpha=0.8)
        ax3.bar(x_pos + width/2, service_importance, width, label='Service Quality', alpha=0.8)
        ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax3.set_ylabel('Percentage of Mediated Effect (%)', fontsize=11)
        ax3.set_xlabel('Treatment Variable', fontsize=11)
        ax3.set_title('Relative Importance: Food vs Service', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(treatment_names, rotation=45, ha='right')
        ax3.legend()
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: R-squared comparison
        ax4 = fig.add_subplot(gs[2, 0])
        
        r2_total = [r['r2_total'] for r in self.results.values()]
        r2_mediated = [r['r2_mediated'] for r in self.results.values()]
        
        x_pos = np.arange(len(treatment_names))
        width = 0.35
        
        ax4.bar(x_pos - width/2, r2_total, width, label='Without Mediators', alpha=0.8)
        ax4.bar(x_pos + width/2, r2_mediated, width, label='With Mediators', alpha=0.8)
        
        ax4.set_ylabel('R² (Variance Explained)', fontsize=11)
        ax4.set_xlabel('Treatment Variable', fontsize=11)
        ax4.set_title('Model Fit: With vs Without Mediators', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(treatment_names, rotation=45, ha='right')
        ax4.legend()
        ax4.set_ylim(0, max(r2_mediated) * 1.2 if r2_mediated else 0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Path diagram for first treatment (example)
        ax5 = fig.add_subplot(gs[2, 1])
        
        if treatment_names:
            first_treatment = treatment_names[0]
            res = self.results[first_treatment]
            
            # Simple path diagram representation
            ax5.text(0.5, 0.9, first_treatment, ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            ax5.text(0.2, 0.5, 'Food\nQuality', ha='center', va='center',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            
            ax5.text(0.8, 0.5, 'Service\nQuality', ha='center', va='center',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            
            ax5.text(0.5, 0.1, 'Overall\nRating', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            
            # Arrows and labels
            # Treatment to Food
            ax5.annotate('', xy=(0.2, 0.55), xytext=(0.45, 0.85),
                        arrowprops=dict(arrowstyle='->', lw=2, color='green'))
            ax5.text(0.3, 0.72, f'a1={res["path_a1"]:.3f}', fontsize=9, color='green')
            
            # Treatment to Service
            ax5.annotate('', xy=(0.8, 0.55), xytext=(0.55, 0.85),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            ax5.text(0.7, 0.72, f'a2={res["path_a2"]:.3f}', fontsize=9, color='red')
            
            # Food to Rating
            ax5.annotate('', xy=(0.4, 0.15), xytext=(0.25, 0.45),
                        arrowprops=dict(arrowstyle='->', lw=2, color='green'))
            ax5.text(0.3, 0.28, f'b1={res["path_b1"]:.3f}', fontsize=9, color='green')
            
            # Service to Rating
            ax5.annotate('', xy=(0.6, 0.15), xytext=(0.75, 0.45),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            ax5.text(0.7, 0.28, f'b2={res["path_b2"]:.3f}', fontsize=9, color='red')
            
            # Direct effect
            ax5.annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.85),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', linestyle='--'))
            ax5.text(0.55, 0.5, f"c'={res['direct_effect']:.3f}", fontsize=9, color='blue')
            
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.axis('off')
            ax5.set_title(f'Path Diagram: {first_treatment}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Mediation plots saved to '{save_path}'")
        
        return fig





def run_mediation_pipeline(df, treatments=None):
    """
    Run complete mediation analysis for multiple treatment variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Restaurant ratings data
    treatments : list of str, optional
        List of treatment variables to analyze
        Default: ['age_group', 'activity', 'personality', 'User_cuisine']
    """
    
    print("="*80)
    print("MEDIATION ANALYSIS PIPELINE")
    print("Research Question: How much of the overall rating is explained by")
    print("                   food quality vs service quality?")
    print("="*80)
    
    # Default treatments if none specified
    if treatments is None:
        treatments = ['age_group', 'activity', 'personality', 'User_cuisine']

    
    # Initialize mediation analysis
    analyzer = MediationAnalysis(df)
    
    # Run analysis for each treatment
    print(f"\n🔬 Analyzing {len(treatments)} treatment variables...")
    
    for treatment in treatments:
        if treatment in df.columns:
            analyzer.analyze_treatment(treatment)
        else:
            print(f"\n⚠ Column '{treatment}' not found in data - skipping")
    
    # Generate summary
    summary_df = analyzer.create_summary_table()
    
    # Visualize results
    if len(analyzer.results) > 0:
        analyzer.plot_mediation_results()
        
        # Save summary to CSV
        if summary_df is not None:
            summary_df.to_csv('mediation_summary.csv', index=False)
            print("\n✓ Summary saved to 'mediation_summary.csv'")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if len(analyzer.results) > 0:
        # Find treatment with highest food mediation
        food_importance = {name: res['food_relative'] 
                          for name, res in analyzer.results.items()
                          if abs(res['total_indirect']) > 0.001}
        
        if food_importance:
            max_food = max(food_importance.items(), key=lambda x: x[1])
            print(f"\n✓ Food quality is MOST important for: {max_food[0]}")
            print(f"  → Explains {max_food[1]:.1%} of the mediated effect")
            
            min_food = min(food_importance.items(), key=lambda x: x[1])
            print(f"\n✓ Service quality is MOST important for: {min_food[0]}")
            print(f"  → Food explains only {min_food[1]:.1%} of mediated effect")
            print(f"  → Service explains {(1-min_food[1]):.1%} of mediated effect")
        
        # Find treatment with highest total mediation
        total_mediation = {name: abs(res['prop_mediated_total'])
                          for name, res in analyzer.results.items()}
        
        if total_mediation:
            max_mediated = max(total_mediation.items(), key=lambda x: x[1])
            print(f"\n✓ Most mediated through food/service: {max_mediated[0]}")
            print(f"  → {max_mediated[1]:.1%} of total effect goes through mediators")
            
            min_mediated = min(total_mediation.items(), key=lambda x: x[1])
            print(f"\n✓ Least mediated (most direct): {min_mediated[0]}")
            print(f"  → Only {min_mediated[1]:.1%} of effect goes through mediators")
    
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
When interpreting results:

1. PROPORTION MEDIATED:
   - High % → Effect works mainly through food/service quality
   - Low % → Effect is direct, not through mediators

2. RELATIVE IMPORTANCE:
   - Food > 50% → Food quality is more important pathway
   - Service > 50% → Service quality is more important pathway

3. R² INCREASE:
   - Large increase → Mediators explain a lot of variance
   - Small increase → Mediators don't add much explanatory power

4. PRACTICAL MEANING:
   - If user characteristics affect ratings mainly through food quality,
     focus on matching users to restaurants with appropriate food styles
   - If through service quality, focus on service-related factors
    """)
    
    return analyzer, summary_df