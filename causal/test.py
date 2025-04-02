import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def identify_confounders(X, T, Y, feature_names, significance_threshold=0.05):
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
        
    Returns:
    --------
    list
        List of identified confounders with their associations
    """
    # Flatten T and Y if needed
    if isinstance(T, np.ndarray) and T.ndim > 1:
        T = T[:, 0]  # Use first treatment if multiple
    
    if isinstance(Y, np.ndarray) and Y.ndim > 1:
        Y = Y[:, 0]  # Use first outcome if multiple
    
    results = []
    
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
            results.append({
                'feature': feature_name,
                'treatment_correlation': t_corr,
                'treatment_pvalue': t_pval,
                'outcome_correlation': y_corr,
                'outcome_pvalue': y_pval,
                'confounding_strength': abs(t_corr * y_corr)  # Simple metric for confounding strength
            })
    
    # Sort by confounding strength
    results = sorted(results, key=lambda x: x['confounding_strength'], reverse=True)
    
    return results

def visualize_confounders(confounders, X, T, Y, feature_names):
    """
    Create visualizations to help understand identified confounders.
    
    Parameters:
    -----------
    confounders : list
        List of confounder dictionaries from identify_confounders
    X : array-like
        Feature matrix
    T : array-like
        Treatment variable
    Y : array-like
        Outcome variable
    feature_names : list
        List of feature names
    """
    if not confounders:
        print("No significant confounders identified.")
        return
    
    # Print identified confounders
    print(f"Identified {len(confounders)} potential confounders:")
    for i, conf in enumerate(confounders):
        print(f"{i+1}. {conf['feature']}")
        print(f"   Treatment correlation: {conf['treatment_correlation']:.4f} (p={conf['treatment_pvalue']:.4f})")
        print(f"   Outcome correlation: {conf['outcome_correlation']:.4f} (p={conf['outcome_pvalue']:.4f})")
        print(f"   Confounding strength: {conf['confounding_strength']:.4f}")
    
    # Create visualization of top confounders
    top_n = min(5, len(confounders))
    
    # Prepare data for visualization
    vis_data = []
    for conf in confounders[:top_n]:
        feature_idx = feature_names.index(conf['feature'])
        feature_values = X[:, feature_idx]
        
        for i in range(len(feature_values)):
            vis_data.append({
                'Confounder': conf['feature'],
                'Confounder Value': feature_values[i],
                'Treatment': T[i] if isinstance(T, np.ndarray) and T.ndim == 1 else T[i, 0],
                'Outcome': Y[i] if isinstance(Y, np.ndarray) and Y.ndim == 1 else Y[i, 0]
            })
    
    vis_df = pd.DataFrame(vis_data)
    
    # Create visualizations
    for conf in confounders[:top_n]:
        conf_data = vis_df[vis_df['Confounder'] == conf['feature']]
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Confounder vs Treatment
        plt.subplot(1, 2, 1)
        sns.regplot(x='Confounder Value', y='Treatment', data=conf_data)
        plt.title(f"{conf['feature']} vs Treatment")
        
        # Plot 2: Confounder vs Outcome
        plt.subplot(1, 2, 2)
        sns.regplot(x='Confounder Value', y='Outcome', data=conf_data)
        plt.title(f"{conf['feature']} vs Outcome")
        
        plt.tight_layout()
        plt.show()
    
    # Create confounding strength plot
    plt.figure(figsize=(10, 6))
    confounder_names = [conf['feature'] for conf in confounders[:top_n]]
    confounder_strengths = [conf['confounding_strength'] for conf in confounders[:top_n]]
    
    plt.barh(confounder_names, confounder_strengths)
    plt.xlabel('Confounding Strength')
    plt.title('Top Confounders by Strength')
    plt.tight_layout()
    plt.show()

# Example usage
"""
# Identify confounders
confounders = identify_confounders(
    data_prep['X'],
    data_prep['T'],
    data_prep['Y'],
    data_prep['feature_names']
)

# Visualize confounders
visualize_confounders(
    confounders,
    data_prep['X'],
    data_prep['T'],
    data_prep['Y'],
    data_prep['feature_names']
)
"""