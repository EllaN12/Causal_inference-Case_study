#%%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from causal_analysis_module.analysis import data_preprocessing
from causal_analysis_module.prepare_data import prepare_for_causal_analysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas_flavor as pf


treatment_lst = ['food_rating', 'service_rating', 'age_group', 'activity', 'personality', 'User_cuisine']

# Run Pre-processing function
data_prep = prepare_for_causal_analysis(
    df = data_preprocessing(),
    treatment_vars= treatment_lst,
    outcome_var= 'rating'
)

print (data_prep)
# create model

model  = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, max_depth=10),
    model_t=RandomForestRegressor(n_estimators=100, max_depth=10),
    criterion= "mse",
    n_estimators= 100,
    honest= True,
    inference= True
)


# Identify Confounders
@pf.register_dataframe_method
def identify_confounders(X, T, Y, feature_names, significance_threshold=0.05,
                         confounding_strength_threshold=0.4,
                         correlation_threshold=0.5):
    """
    Identify potential confounders by finding variables that are
    significantly associated with both treatment and outcome.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    T : array-like
        Treatment variable (flattened to 1D if needed)
    Y : array-like
        Outcome variable (flattened to 1D if needed)
    feature_names : list
        List of feature names
    significance_threshold : float, default=0.05
        P-value threshold for significance
    confounding_strength_threshold : float, default=0.4
        Minimum confounding strength to include (40%)
    correlation_threshold : float, default=0.5
        Absolute correlation threshold (50%)
        
    Returns:
    --------
    list
        List of identified confounders with their associations
    dict
        Dictionary containing indices of strong confounders
    """
    # Flatten T and Y if needed
    if isinstance(T, np.ndarray) and T.ndim > 1:
        T = T[:, 0]  # Use first treatment if multiple
    
    if isinstance(Y, np.ndarray) and Y.ndim > 1:
        Y = Y[:, 0]  # Use first outcome if multiple
    
    results = []
    strong_confounder_indices = []
    
    for i, feature_name in enumerate(feature_names):
        if i >= X.shape[1]:
            continue  # Skip if feature index is out of bounds
            
        feature = X[:, i]
        
        # Calculate correlation with treatment
        t_corr, t_pval = stats.pearsonr(feature, T)
        
        # Calculate correlation with outcome
        y_corr, y_pval = stats.pearsonr(feature, Y)
        
        # Check if significantly associated with both
        is_confounder = (t_pval < significance_threshold) and (y_pval < significance_threshold)
        
        if is_confounder:
            confounding_strength = abs(t_corr * y_corr)
            
            confounder_info = {
                'feature': feature_name,
                'treatment_correlation': t_corr,
                'treatment_pvalue': t_pval,
                'outcome_correlation': y_corr,
                'outcome_pvalue': y_pval,
                'confounding_strength': confounding_strength,
                'index': i
            }
            
            results.append(confounder_info)
            
            # Apply the new filtering criteria
            if (confounding_strength > confounding_strength_threshold and 
                (abs(t_corr) >= correlation_threshold or abs(y_corr) >= correlation_threshold)):
                strong_confounder_indices.append(i)
    
    # Sort by confounding strength
    results = sorted(results, key=lambda x: x['confounding_strength'], reverse=True)
    
    return results, strong_confounder_indices

@pf.register_dataframe_method
def apply_confounder_filter_to_model(X, T, Y, feature_names, train_test_split_ratio=0.8, random_state=42):
    """
    Apply confounder identification and filtering, then build a CausalForestDML model.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    T : array-like
        Treatment variable
    Y : array-like
        Outcome variable
    feature_names : list
        List of feature names
    train_test_split_ratio : float, default=0.8
        Ratio for train/test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    model
        Fitted CausalForestDML model
    dict
        Dictionary with model performance metrics
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Identify confounders
    confounders_info, strong_indices = identify_confounders(
        X, T, Y, feature_names,
        confounding_strength_threshold=0.4,
        correlation_threshold=0.5
    )
    
    print(f"Identified {len(confounders_info)} total confounders")
    print(f"Found {len(strong_indices)} strong confounders meeting all criteria:")
    
    # Print the strong confounders
    strong_confounders = [cf for cf in confounders_info if cf['index'] in strong_indices]
    for cf in strong_confounders:
        print(f"  - {cf['feature']}: confounding_strength={cf['confounding_strength']:.4f}, " +
              f"treatment_correlation={cf['treatment_correlation']:.4f}, " +
              f"outcome_correlation={cf['outcome_correlation']:.4f}")
    
    # If no strong confounders, use all confounders
    if not strong_indices:
        print("No strong confounders found meeting all criteria. Using all identified confounders.")
        strong_indices = [cf['index'] for cf in confounders_info]
    
    # Check if treatment is multi-dimensional
    if isinstance(T, np.ndarray) and T.ndim > 1:
        selected_treatment = T[:, 0]  # Use first treatment for simplicity
        print(f"Using first dimension from {T.shape[1]}-dimensional treatment")
    else:
        selected_treatment = T
    
    # Split the data
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, selected_treatment, Y, train_size=train_test_split_ratio, random_state=random_state
    )
    
    # Fit CausalForestDML model with the identified confounders
    cf_model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=random_state),
        model_t=RandomForestRegressor(n_estimators=100, random_state=random_state),
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=10,
        verbose=0,
        random_state=random_state
    )
    
    # Include the strong confounders in X
    cf_model.fit(
        Y_train, 
        T_train,
        X=X_train  # All features including confounders
    )
    
    # Estimate treatment effects
    te_test = cf_model.effect(X_test)
    
    # Calculate confidence intervals
    lower, upper = cf_model.effect_interval(X_test, alpha=0.05)
    
    #
    
    # Return the model and some metrics
    results = {
        'model': cf_model,
        'treatment_effects': te_test,
        'confidence_intervals': (lower, upper),
        'strong_confounders': strong_confounders,
        'all_confounders': confounders_info,
    }
    
    return results


# fit the model 
# Fit the model
model.fit(
    Y= data_prep['Y_train'],
    T= data_prep['T_train'],
    X= data_prep['X_train'],
    W= None
)


# Analyse the treatment effect for treatment model
@pf.register_dataframe_method
def treatment_effect_dataframe(treatment_effects, treatment_name, data_prep, covariates = None):
    
    """_summary_
    Creates a DataFrame with treatment effects and key features for analysis.
    
    Parameters:
    -----------
    treatment_effects : numpy.ndarray
        Array of individual treatment effects from the model
    treatment_name : str or list
        Name(s) of the treatment variable(s)
    data_prep : dict
        Dictionary containing test data and other information
    covariates : list, optional
        List of covariates to include in the DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with treatment effects and related data
    """

    
    # convert treatment to list
    treatment_name = treatment_name
    
    # Create the base DataFrame with treatment effects
    if len(treatment_name) == 1:
        # Single treatment case
        df = pd.DataFrame({
            'treatment_effect': treatment_effects,
            'treatment_name': treatment_name[0]
        })
    else:
        # Multiple treatments case (assuming treatment_effects has multiple columns)
        df = pd.DataFrame()
        
        if treatment_effects.ndim > 1 and treatment_effects.shape[1] == len(treatment_name):
            for i, name in enumerate(treatment_name):
                df[f'{name}_effect'] = treatment_effects[:, i]
                
        else:
            # If treatment_effects is just a 1D array despite multiple treatment names
            df['treatment_effect'] = treatment_effects
            df['treatment_name'] = 'combined_treatment'
            
        # Add Test data if available
        
        if data_prep is not None:
            # Add actual outcome if available
            
            if 'Y_test' in data_prep:
                df['outcome'] = data_prep['Y_test']
                
        # Add actual treatment values if available
            if 'T_test' in data_prep:
                T_test = data_prep['T_test']
                
            # Add actual treatment values if available
                if isinstance(T_test, np.ndarray) and T_test.ndim > 1 and T_test.shape[1] == len(treatment_name):
                # Multiple treatments
                    for i, name in enumerate(treatment_name):
                        df[f'actual_{name}'] = T_test[:, i]
                    
                else:
                    # single treatement
                    if len(treatment_name) == 1:
                        df[f'actual_{treatment_name[0]}'] = ['T_test']
                    else:
                        # # If T_test is 1D but we have multiple treatment names
                        df['treatment_name'] = T_test
                        
        # Add additional analysis columns
        if 'treatment_effect' in df.columns:
            df['positive_effect'] = df['treatment_effect'] >0
            df['effect_magnitude'] = np.abs(df['treatment_effect'])
            
            # Categorize Effect Size
            df['effect_category'] = pd.cut(
            df['treatment_effect'], 
            bins=[-float('inf'), -0.5, -0.1, 0.1, 0.5, float('inf')],
            labels=['Strong Negative', 'Moderate Negative', 'Negligible', 'Moderate Positive', 'Strong Positive']
        )
        
        return df
 
 
treatment_effects  = model.effect(data_prep['X_test'])
lb, ub = model.effect_interval(data_prep['X_test'], alpha= 0.05)
   
   
treatment_lst = ['food_rating', 'service_rating', 'age_group', 'activity', 'personality', 'User_cuisine']
effects_df = treatment_effect_dataframe(
    treatment_effects = treatment_effects,
    treatment_name = treatment_lst,
    data_prep = data_prep
)
# %%
# Estimate Treatment Effects 

# Count of effect categories
print("\nEffect Categories:")
print(effects_df['effect_category'].value_counts())

import matplotlib.pyplot as plt
import numpy as np

# Define your data
categories = ['Strong Negative', 'Moderate Negative', 'Negligible', 'Moderate Positive', 'Strong Positive']
counts = [1488, 539, 457, 1066, 2762]

# Create a list of tuples (category, count, color)
colors = ['#d73027', '#fc8d59', '#ffffbf', '#91cf60', '#1a9850']
data = list(zip(categories, counts, colors))

# Sort by count in descending order
sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
sorted_categories, sorted_counts, sorted_colors = zip(*sorted_data)

# Calculate percentages
total = sum(counts)
percentages = [count/total*100 for count in sorted_counts]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar chart with sorted data
bars = ax.bar(sorted_categories, sorted_counts, color=sorted_colors)

# Customize the plot
ax.set_title('Treatment Effect Categories Distribution', fontsize=15)
ax.set_xlabel('Effect Category', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.tick_params(axis='x', rotation=45)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,}\n({height/total*100:.1f}%)',
            ha='center', va='bottom')

# Add summary statistics
positive_counts = counts[3] + counts[4]  # Moderate Positive + Strong Positive
negative_counts = counts[0] + counts[1]  # Strong Negative + Moderate Negative
neutral_counts = counts[2]  # Negligible

positive_pct = positive_counts / total * 100
negative_pct = negative_counts / total * 100
neutral_pct = neutral_counts / total * 100

plt.figtext(0.5, 0.01, 
           f'Total observations: {total:,}\n'
           f'Positive effects: {positive_counts:,} ({positive_pct:.1f}%), '
           f'Negative effects: {negative_counts:,} ({negative_pct:.1f}%), '
           f'Neutral: {neutral_counts:,} ({neutral_pct:.1f}%)', 
           ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Add grid lines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Make layout tight
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Print the sorted counts
print("Categories sorted by count (descending):")
for category, count, percentage in zip(sorted_categories, sorted_counts, percentages):
    print(f"{category}: {count:,} ({percentage:.1f}%)")

# Save the figure (optional)
plt.savefig('treatment_effect_histogram_sorted.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


ate = np.mean(treatment_effects)
    
# Basic statistics
print("\nSummary Statistics:")
print(effects_df['treatment_effect'].describe())
    
@pf.register_dataframe_method
def effect_graph():
    effect_category_counts = effects_df['effect_category'].value_counts().reset_index()
    effect_category_counts.columns = ['effect_category', 'count']
    df = effect_category_counts
    # Sort data by count in descending order for better visualization
    df = df.sort_values('count', ascending=False)
    # Create a colormap based on the effect categories
    colors = ['#2ca02c', '#d62728', '#7fc97f', '#e08214', '#999999']  # green, red, light green, orange, gray
    # Create the bar plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df['effect_category'], df['count'], color=colors)

    # Add count values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{height:,}', ha='center', fontsize=12)

        # Add labels and title
    plt.xlabel('Effect Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Distribution of Effect Categories', fontsize=16, fontweight='bold')

    # Add grid lines for better readability
    plt.grid(axis='y', alpha=0.3)

    # Add total count in the figure
    total_count = df['count'].sum()
    plt.figtext(0.5, 0.01, f'Total Count: {total_count:,}', ha='center', fontsize=12)

        # Improve layout
    plt.tight_layout(pad=3)

    # Display the plot
    plt.show()







