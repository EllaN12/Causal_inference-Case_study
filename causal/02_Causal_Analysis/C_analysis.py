#%%
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np
from IPython.display import Image
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
from graphviz import Digraph
import pydot
import io
import tempfile
from causal_analysis_module.analysis import data_preprocessing
from dowhy import CausalModel


# %%

#import clean dataset:
# Load the DataFrame from the pickle file
dir_path = os.getcwd()
df = pd.read_pickle(os.path.join(dir_path, "01_Data_analysis", "final_data.pkl" ))

df.info()
print(df.columns.tolist())


df['other_services'].unique()


numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_columns


category_columns = df.select_dtypes(include = ['object']).columns.tolist()
category_columns

#%%
print(category_columns)

##Feature Engineering
# Create other features In order to determine relationships between data points, outline relationships between data points
#to establish the causality

# Patrons- Restaurant Distance
def calculate_distance(row):
        """Calculate distance between user and restaurant"""
        user_loc = (row['p.latitude'], row['p.longitude'])
        rest_loc = (row['latitude'], row['longitude'])
        return geodesic(user_loc, rest_loc).kilometers
    


df['patrons_restaurant_distance'] = df.apply(calculate_distance, axis=1)  
        
# Restaurant Clsuters Feature based on geolocation  
coords = df[['latitude', 'longitude']].values
df['location_cluster'] = KMeans(n_clusters=5).fit_predict(coords)
 
# Create age groups from birth_year
current_year = 2025  # Adjust as needed
df['age'] = current_year - df['birth_year']
df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 25, 35, 50, 100],
                              labels=['18-25', '26-35', '36-50', '50+'])

# Add cuisine matching feature
def calculate_cuisine_match(row):
    user_cuisines = row['User_cuisine'].split(',') if isinstance(row['User_cuisine'], str) else []
    rest_cuisines = row['restaurant_specialty'].split(',') if isinstance(row['restaurant_specialty'], str) else []
    
    # Calculate matching score
    if user_cuisines and rest_cuisines:
        matches = len(set(user_cuisines) & set(rest_cuisines))
        total = len(set(user_cuisines))
        return matches / total if total > 0 else 0
    return 0
    
df['cuisine_match_score'] = df.apply(calculate_cuisine_match, axis=1)

print(df)



#%%
def prepare_data_for_dowhy(data):
    """
    Prepare the dataset for DoWhy analysis by encoding categorical variables
    """
    # Create label encoders for categorical variables
    categorical_columns = [
        'age_group', 'marital_status', 'activity', 'transport',
        'User_cuisine', 'Rcuisine_y', 'accessibility', 'smoking_area',
        'area', 'user_ambience', 'dress_preference'
    ]
    
    encoders = {}
    encoded_data = data.copy()
    
    # Encode categorical variables
    for col in categorical_columns:
        if col in encoded_data.columns:
            encoders[col] = LabelEncoder()
            encoded_data[col] = encoders[col].fit_transform(encoded_data[col].astype(str))
    
    return encoded_data, encoders

prepare_data_for_dowhy(
     data = df
)
#%%


df = data_preprocessing()

def create_causal_graph_spec():
    """
    Create the causal graph specification for DoWhy
    """
    graph = """
    digraph {
        # Direct Effects on Rating
        food_rating -> rating;
        service_rating -> rating;
        age_group -> rating;
        activity -> rating;
        personality -> rating;
        User_cuisine -> rating;
        
        # Demographic Effects
        age_group -> budget;
        age_group -> User_cuisine;
        age_group -> drink_level;
        age_group -> dress_preference;
        
        activity -> budget;
        activity -> transport;
        
        marital_status -> hijos;
        marital_status -> budget;
        marital_status -> user_ambience;
        
        # Restaurant Selection Effects
        transport -> accessibility;
        transport -> area;
        budget -> price;
        
        # Personal Attribute Effects
        weight -> food_rating;
        personality -> food_rating;
        height -> food_rating;
        color -> food_rating;
        hijos -> food_rating;
        
        # Service Rating Influences
        color -> service_rating;
        personality -> service_rating;
        
        # Transport and Accessibility Chain
        transport -> smoker;
        transport -> dress_preference;
        transport -> weight;
    }
    """
    return graph

def create_and_visualize_graph(graph_spec, plot=True, treatment_nodes=None, outcome_node="rating"):
    """
    Creates a directed acyclic graph (DAG) from a graphviz specification and visualizes it.
    Uses pydot instead of pygraphviz for better compatibility.
    
    Args:
        graph_spec: A string containing the graphviz graph specification.
        plot: If True, plots the graph using matplotlib.
        treatment_nodes: List of treatment nodes to highlight in a single color.
        outcome_node: The outcome node to highlight in red.
    
    Returns:
        A networkx.DiGraph object representing the DAG.
    """
    # Default treatment nodes if None provided
    if treatment_nodes is None:
        treatment_nodes = ['food_rating', 'service_rating', 'age_group', 'activity', 
                          'personality', 'User_cuisine']
    
    # Parse with pydot
    graphs = pydot.graph_from_dot_data(graph_spec)
    if not graphs:
        raise ValueError("Could not parse the DOT graph specification")
    
    # Convert to networkx graph
    graph = nx.nx_pydot.from_pydot(graphs[0])
    
    if plot:
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(graph, seed=42)  # Consistent layout
        
        # Define colors for node categories
        treatment_color = 'green'  # Single color for all treatment nodes
        outcome_color = 'red'      # Red for outcome
        other_color = 'skyblue'    # Sky blue for all other nodes
        
        # Create a node list and corresponding color list
        all_nodes = list(graph.nodes())
        node_colors = []
        
        for node in all_nodes:
            if node == outcome_node:
                node_colors.append(outcome_color)
            elif node in treatment_nodes:
                node_colors.append(treatment_color)
            else:
                node_colors.append(other_color)
        
        # Draw the graph with node colors
        nx.draw(graph, pos, 
                nodelist=all_nodes,
                node_color=node_colors,
                with_labels=True, 
                node_size=700, 
                font_size=8,
                arrowstyle='-|>', 
                arrowsize=15)
        
        # Create simplified legend with categories
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=treatment_color, 
                      markersize=10, label='Treatments'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=outcome_color, 
                      markersize=10, label='Outcome'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=other_color, 
                      markersize=10, label='Other Variables')
        ]
        
        # Place legend
        plt.legend(handles=legend_handles,
                  title="Variable Categories",
                  loc='lower right',
                  frameon=True,
                  framealpha=0.8)
        
        plt.title("Causal Graph with Treatment Variables and Outcome")
        plt.tight_layout()
        plt.show()
    
    return graph



create_and_visualize_graph(
    graph_spec= create_causal_graph_spec(),
)



# %%

create_and_visualize_graph(
    graph_spec= create_causal_graph_spec()
)


# Create Causal model

def create_networkx_graph():
    """
    Create a NetworkX DiGraph directly without using DOT.
    This is an alternative approach if the DOT parsing continues to fail.
    """
    G = nx.DiGraph()
    
    # Add nodes
    nodes = [
        'food_rating', 'service_rating', 'rating', 'age_group', 'activity', 'personality', 
        'User_cuisine', 'budget', 'drink_level', 'dress_preference', 'transport', 
        'marital_status', 'hijos', 'user_ambience', 'accessibility', 'area', 'price',
        'weight', 'height', 'color', 'smoker'
    ]
    G.add_nodes_from(nodes)
    
    # Add edges - Direct mapping from your DOT specification
    edges = [
        # Direct Effects on Rating
        ('food_rating', 'rating'), ('service_rating', 'rating'), ('age_group', 'rating'),
        ('activity', 'rating'), ('personality', 'rating'), ('User_cuisine', 'rating'),
        
        # Demographic Effects
        ('age_group', 'budget'), ('age_group', 'User_cuisine'), ('age_group', 'drink_level'),
        ('age_group', 'dress_preference'),
        
        ('activity', 'budget'), ('activity', 'transport'),
        
        ('marital_status', 'hijos'), ('marital_status', 'budget'), ('marital_status', 'user_ambience'),
        
        # Restaurant Selection Effects
        ('transport', 'accessibility'), ('transport', 'area'), ('budget', 'price'),
        
        # Personal Attribute Effects
        ('weight', 'food_rating'), ('personality', 'food_rating'), ('height', 'food_rating'),
        ('color', 'food_rating'), ('hijos', 'food_rating'),
        
        # Service Rating Influences
        ('color', 'service_rating'), ('personality', 'service_rating'),
        
        # Transport and Accessibility Chain
        ('transport', 'smoker'), ('transport', 'dress_preference'), ('transport', 'weight')
    ]
    G.add_edges_from(edges)
    
    return G





def create_causal_models(df, treatment_lst, outcome='rating'):
    """Create causal models for each treatment variable."""
    models = {}

    # Get the graph specification once
    graph_spec = create_networkx_graph()

    # For DoWhy's CausalModel, we can directly pass the DOT string
    for treatment_name in treatment_lst:
        model = CausalModel(
            data=df,
            treatment=treatment_name,
            outcome=outcome,
            graph=graph_spec  # Pass the DOT string directly
        )
        models[treatment_name] = model

    return models


treatment_lst = ['food_rating', 'service_rating', 'age_group', 'activity', 'personality', 'User_cuisine']


models = create_causal_models(
    df = data_preprocessing(),
    treatment_lst= treatment_lst, 
)
print (models)


# Generating estimands:
def generate_causal_estimands(models):
    """
    Generate and return causal estimands for all treatment variables.
    
    Parameters:
    -----------
    models : dict
        Dictionary of CausalModel objects with treatment variables as keys
        
    Returns:
    --------
    dict
        Dictionary with treatment variables as keys and identified estimands as values
    """
    estimands = {}
    
    for treatment_name, model in models.items():
        try:
            # Check whether causal effect is identified and return target estimand
            identified_estimand = model.identify_effect()
            estimands[treatment_name] = identified_estimand
            print(f"✓ Successfully identified estimand for treatment: {treatment_name}")
        except Exception as e:
            print(f"✗ Error identifying estimand for treatment '{treatment_name}': {str(e)}")
            estimands[treatment_name] = None
    
    return estimands


# Create dictionary  of estimamnds based on identified treatment variables
treatment_lst = ['food_rating', 'service_rating', 'age_group', 'activity', 'personality', 'User_cuisine']
estimands = generate_causal_estimands(models)
    


#Quantify the estmates the target effect based on treatment variables
def estimate_causal_effects(models, estimands, method_names=None):
    """
    Estimate causal effects for all treatment variables using multiple methods.
    
    Parameters:
    -----------
    models : dict
        Dictionary of CausalModel objects with treatment variables as keys
    estimands : dict
        Dictionary with treatment variables as keys and identified estimands as values
    method_names : list, optional
        List of estimation method names to use. If None, will use a default set of methods.
        
    Returns:
    --------
    dict
        Dictionary with treatment variables as keys and a nested dictionary of estimates by method as values
    """
    # Default estimation methods suitable for this causal graph structure
    if method_names is None:
        method_names = [
            "backdoor.propensity_score_matching",
            "backdoor.propensity_score_stratification", 
            "backdoor.propensity_score_weighting",
            "backdoor.linear_regression",
            "iv.instrumental_variable"  # Only if you have valid instruments
        ]
    
    estimates = {}
    
    for treatment_name, model in models.items():
        if estimands[treatment_name] is None:
            print(f"Skipping estimation for {treatment_name} as no valid estimand was identified")
            continue
            
        treatment_estimates = {}
        
        for method_name in method_names:
            try:
                # For IV methods, we should only try if we have valid instruments
                if method_name.startswith("iv.") and not has_valid_instruments(model, treatment_name):
                    print(f"Skipping {method_name} for {treatment_name} as no valid instruments were identified")
                    continue
                
                print(f"Estimating effect of {treatment_name} using {method_name}...")
                estimate = model.estimate_effect(
                    estimands[treatment_name],
                    method_name=method_name
                )
                
                # Store the estimate
                treatment_estimates[method_name] = estimate
                print(f"  ✓ {method_name}: Effect = {estimate.value}")
                
            except Exception as e:
                print(f"  ✗ Error with {method_name} for {treatment_name}: {str(e)}")
        
        estimates[treatment_name] = treatment_estimates
    
    return estimates

estimate_causal_effects(
    models= models,
    estimands= estimands
)

estimate_causal_effects(
    models= models['food_rating'],
    estimands= estimands['food_rating']
)




# Try Analysis One at a time:
def analyze_food_rating_effect(df):
    """
    Analyze the causal effect of food_rating on overall rating.
    
    Parameters:
    -----------
    df : DataFrame
        The preprocessed data
        
    Returns:
    --------
    dict
        Results of causal effect estimation for food_rating
    """
    # Step 1: Create the causal model just for food_rating
    model = create_causal_models(
        df=df,
        treatment_lst=['food_rating']  # Only food_rating
    )['food_rating']  # Extract the single model
    
    # Step 2: Generate the estimand
    try:
        estimand = model.identify_effect()
        print(f"✓ Successfully identified estimand for food_rating")
    except Exception as e:
        print(f"✗ Error identifying estimand for food_rating: {str(e)}")
        return None
    
    # Step 3: Estimate causal effects with multiple methods
    methods = [
        "backdoor.propensity_score_matching",
        "backdoor.propensity_score_stratification", 
        "backdoor.propensity_score_weighting",
        "backdoor.linear_regression"
    ]
    
    results = {
        'estimand': estimand,
        'estimates': {}
    }
    
    print("\nEstimating causal effect of food_rating on rating:")
    print("------------------------------------------------")
    
    for method in methods:
        try:
            estimate = model.estimate_effect(estimand, method_name=method)
            results['estimates'][method] = estimate
            
            # Extract the effect value and confidence interval if available
            effect_value = estimate.value
            ci_low, ci_high = None, None
            if hasattr(estimate, 'confidence_intervals'):
                ci = estimate.confidence_intervals
                if ci is not None and len(ci) >= 2:
                    ci_low, ci_high = ci[0], ci[1]
            
            # Print the results
            ci_str = f" (95% CI: [{ci_low:.4f}, {ci_high:.4f}])" if ci_low is not None else ""
            print(f"  ✓ {method}: Effect = {effect_value:.4f}{ci_str}")
            
        except Exception as e:
            print(f"  ✗ {method}: Error - {str(e)}")
    
    return results

food_rating_results = analyze_food_rating_effect(df = data_preprocessing())

def visualize_food_rating_effects(results):
    """
    Create a visualization of the causal effect estimates for food_rating.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_food_rating_effect function
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if results is None or 'estimates' not in results:
        print("No valid results to visualize")
        return
        
    estimates = results['estimates']
    if not estimates:
        print("No valid estimates to visualize")
        return
        
    # Extract method names and effect values
    methods = []
    effects = []
    errors = []
    
    for method, estimate in estimates.items():
        # Format the method name for better readability
        method_name = method.split('.')[-1].replace('_', ' ').title()
        methods.append(method_name)
        
        # Get the effect value
        effect = estimate.value
        effects.append(effect)
        
        # Get confidence interval if available
        error = 0
        if hasattr(estimate, 'confidence_intervals'):
            ci = estimate.confidence_intervals
            if ci is not None and len(ci) >= 2:
                error = (ci[1] - ci[0]) / 2
        errors.append(error)
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(methods))
    
    plt.barh(y_pos, effects, xerr=errors, align='center', alpha=0.7, 
             color='skyblue', capsize=5)
    
    plt.yticks(y_pos, methods)
    plt.xlabel('Causal Effect of Food Rating on Overall Rating')
    plt.title('Estimated Causal Effect of Food Rating by Method')
    
    # Add a vertical line at x=0 for reference
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add effect values as text
    for i, effect in enumerate(effects):
        plt.text(effect + 0.01, i, f'{effect:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('food_rating_causal_effect.png')
    plt.show()
    
    return plt

visualize_food_rating_effects(results=
                              food_rating_results)

def run_food_rating_analysis(df):
    """
    Run the complete analysis for food_rating causal effect.
    
    Parameters:
    -----------
    df : DataFrame
        The preprocessed data
    """
    # Analyze the causal effect
    results = analyze_food_rating_effect(df)
    
    if results:
        # Visualize the results
        visualize_food_rating_effects(results)
        
        # Provide a textual interpretation
        print("\nInterpretation of Food Rating Causal Effect:")
        print("-------------------------------------------")
        
        # Calculate average effect across methods
        effects = [est.value for est in food_rating_results['estimates'].values()]
        avg_effect = sum(effects) / len(effects)
        
        print(f"The average causal effect of food rating on overall rating is {avg_effect:.4f}.")
        print(f"This means that, on average, a one-unit increase in food rating")
        print(f"causes the overall rating to change by {avg_effect:.4f} units,")
        print(f"holding all other factors constant.")
        
        if avg_effect > 0:
            print("\nThis positive effect suggests that improving food quality")
            print("is an effective strategy for improving overall customer satisfaction.")
        elif avg_effect < 0:
            print("\nThis negative effect is unexpected and warrants further investigation.")
        else:
            print("\nThis negligible effect suggests that food rating may not be")
            print("a key driver of overall satisfaction in this dataset.")
    
    return results



food_rating_results = run_food_rating_analysis(df = data_preprocessing()

)

# Stress Test using Instrumental variable analysis
def get_potential_instruments(treatment_name):
    """
    Get potential instrumental variables for a specific treatment.
    
    Parameters:
    -----------
    treatment_name : str
        The name of the treatment variable
        
    Returns:
    --------
    list
        List of potential instrumental variables
    """
    # Define potential instruments for each treatment based on the causal graph
    instrument_map = {
        'food_rating': ['weight', 'height', 'color', 'hijos', 'marital_status', 'transport'], 
        'User_cuisine': ['age_group', 'transport', 'weight']
    }

food_rating = get_potential_instruments('food_rating')



def run_iv_estimation_for_treatment(df, treatment_name, outcome='rating'):
    """
    Run instrumental variable estimation for a specific treatment.
    
    Parameters:
    -----------
    df : DataFrame
        The preprocessed data
    treatment_name : str
        The name of the treatment variable to analyze
    outcome : str, optional
        The outcome variable (default: 'rating')
        
    Returns:
    --------
    dict
        Dictionary containing the model, estimand, and IV estimate
    """
    # Step 1: Create a causal model just for this treatment
    model = create_causal_models(
        df=df,
        treatment_lst=[treatment_name],
    )[treatment_name]  # Extract the single model
    
    # Step 2: Identify the estimand
    try:
        estimand = model.identify_effect()
        print(f"✓ Successfully identified estimand for {treatment_name}")
    except Exception as e:
        print(f"✗ Error identifying estimand for {treatment_name}: {str(e)}")
        return {
            'treatment': treatment_name,
            'status': 'Failed to identify estimand',
            'error': str(e)
        }
    
    # Step 3: Determine potential instruments based on the treatment
    instruments = get_potential_instruments(treatment_name)
    
    if not instruments:
        print(f"✗ No potential instruments identified for {treatment_name}")
        return {
            'treatment': treatment_name,
            'estimand': estimand,
            'status': 'No instruments available'
        }
    
    print(f"Potential instruments for {treatment_name}: {', '.join(instruments)}")
    
    # Step 4: Attempt IV estimation
    try:
        # Set up IV method parameters
        iv_params = {
            "method_name": "iv.instrumental_variable",
            "method_params": {
                "iv_instruments": instruments
            }
        }
        
        # Estimate effect using IV
        iv_estimate = model.estimate_effect(
            estimand,
            **iv_params
        )
        
        print(f"✓ IV estimation successful for {treatment_name}")
        print(f"  Effect estimate: {iv_estimate.value:.4f}")
        
        if hasattr(iv_estimate, 'confidence_intervals') and iv_estimate.confidence_intervals is not None:
            ci_low, ci_high = iv_estimate.confidence_intervals
            print(f"  95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
        
        # Check if IV estimate is statistically significant
        is_significant = False
        if hasattr(iv_estimate, 'test_statistic') and hasattr(iv_estimate, 'p_value'):
            print(f"  Test statistic: {iv_estimate.test_statistic:.4f}")
            print(f"  p-value: {iv_estimate.p_value:.4f}")
            is_significant = iv_estimate.p_value < 0.05
        
        return {
            'treatment': treatment_name,
            'estimand': estimand,
            'iv_estimate': iv_estimate,
            'instruments_used': instruments,
            'status': 'Success',
            'is_significant': is_significant
        }
        
    except Exception as e:
        print(f"✗ Error in IV estimation for {treatment_name}: {str(e)}")
        return {
            'treatment': treatment_name,
            'estimand': estimand,
            'status': 'Failed IV estimation',
            'error': str(e),
            'instruments_tried': instruments
        }

run_iv_estimation_for_treatment(
    df = data_preprocessing(),
    treatment_name= 'food_rating',
    
)