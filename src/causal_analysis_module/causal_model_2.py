#%%
import os
import os
from bs4 import BeautifulSoup as soup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

#for causal_grapgh
from causalgraphicalmodels import CausalGraphicalModel
import networkx as nx
from dowhy import CausalModel
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
import logging, sys
import pandas_flavor as pf
import pandas as pd
from causal_analysis_module.analysis import data_preprocessing
from causal_analysis_module.causal_graph import webscrape_report, causal_digraph, create_causal_visualization


# %%

create_causal_visualization(
    cgm = (causal_digraph(
        correlation_dict=(webscrape_report())
    ))
)




df = data_preprocessing()


def causal_digraph_for_dowhy(correlation_dict: Dict[str, str]) -> nx.DiGraph:
    """
    Creates a causal graph for use with DoWhy.
    
    Args:
        correlation_dict: A dictionary of correlations, where keys are source variables
                          and values are comma-separated target variables.
    
    Returns:
        A networkx DiGraph object ready for use with DoWhy's CausalModel.
    """
    def parse_relationships_dict(correlation_dict: Dict[str, str]) -> List[Tuple[str, str]]:
        """Parse the relationships dictionary to create edges with error handling."""
        
        additional_edges = []

        if not correlation_dict:
            print("Warning: Empty correlation dictionary provided")
            return additional_edges

        for source, targets in correlation_dict.items():
            if not targets:
                print(f"Warning: No targets found for source '{source}'")
                continue

            try:
                target_list = [t.strip() for t in targets.split(',') if t.strip()]

                for target in target_list:
                    if target:
                        additional_edges.append((str(source), str(target)))

            except Exception as e:
                print(f"Error processing source '{source}': {str(e)}")
                continue

        print(f"Successfully created {len(additional_edges)} edges from correlation dictionary")
        return additional_edges

    # Base causal relationships
    base_edges = [
        # Direct Effects on Rating
        ('food_rating', 'rating'),
        ('service_rating', 'rating'),
        ('age_group', 'rating'),
        ('activity', 'rating'),
        ('personality', 'rating'),
        ('User_cuisine', 'rating'),
        
        # Demographic Effects
        ('age_group', 'budget'),
        ('age_group', 'User_cuisine'),
        ('age_group', 'drink_level'),
        ('age_group', 'dress_preference'),
        ('activity', 'budget'),
        ('activity', 'transport'),
        ('marital_status', 'hijos'),
        ('marital_status', 'budget'),
        ('marital_status', 'user_ambience'),
        
        # Restaurant Selection Effects
        ('transport', 'accessibility'),
        ('transport', 'area'),
        ('budget', 'price'),
        
        # Personal Attribute Effects
        ('weight', 'food_rating'),
        ('personality', 'food_rating'),
        ('height', 'food_rating'),
        ('color', 'food_rating'),
        ('hijos', 'food_rating'),
        
        # Service Rating Influences
        ('color', 'service_rating'),
        ('personality', 'service_rating'),
        
        # Transport and Accessibility Chain
        ('transport', 'smoker'),
        ('transport', 'dress_preference'),
        ('transport', 'weight')
    ]

    try:
        correlation_edges = parse_relationships_dict(correlation_dict)

        # Create a DiGraph directly
        G = nx.DiGraph()
        G.add_edges_from(base_edges)
        logging.info(f"Added {len(base_edges)} base edges")

        if correlation_edges:
            G.add_edges_from(correlation_edges)
            logging.info(f"Added {len(correlation_edges)} correlation edges")

        # Log graph information
        logging.info(f"Created causal graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        
        return G  # Return the DiGraph directly

    except Exception as e:
        logging.error(f"Error creating causal graph: {str(e)}")
        # Fallback: Basic graph with only base edges
        G = nx.DiGraph()
        G.add_edges_from(base_edges)
        logging.warning("Returning fallback causal graph with only base edges")
        return G.dag






def run_causal_analysis(df: pd.DataFrame):
    """
    Run causal analysis for multiple treatment variables.
    
    Args:
        df: DataFrame with all necessary variables
        
    Returns:
        Dictionary with results for each treatment variable
    """
    # List of treatment variables to analyze
    treatment_lst = ['food_rating', 'service_rating', 'age_group', 'activity', 'personality', 'User_cuisine']
    
    # Create a dictionary to store model results for each treatment
    model_results = {}
    
    # Get correlation dictionary from your webscrape_report function
    correlation_dict = webscrape_report()
    
    # Get the networkx DiGraph directly
    nx_graph = causal_digraph_for_dowhy(correlation_dict)
    
    # Iterate through each treatment variable
    for treatment_name in treatment_lst:
        print(f"Analyzing treatment: {treatment_name}")
        
        # Create a causal model for each treatment variable
        model = CausalModel(
            data=df,
            treatment=treatment_name,
            outcome='rating',
            graph=nx_graph  # Use the DiGraph directly
        )
        
        # Run analysis for this model
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.metalearners.XLearner",
            method_params={"init_params":{"models":"forest"}}
        )
        
        # Store the results
        model_results[treatment_name] = estimate
        
        # Print results for each treatment
        print(f"Treatment: {treatment_name}")
        print(estimate)
        print("-" * 50)
    
    return model_results


run_causal_analysis(df = df)

# Create a causal model from the data and given graph.
model=CausalModel(
        data = df,
        treatment=df['treatment_name'],
        outcome=df['rating'],
        graph= (causal_digraph(
        correlation_dict=(webscrape_report()))
        ))


treatment_lst = ['food_rating', 'service_rating', 'age_group', 'activity', 'personality', 'User_cuisine']


# Create a dictionary to store model results for each treatment
model_results = {}

# Iterate through each treatment variable
for treatment_name in treatment_lst:
    # Create a causal model for each treatment variable
    model = CausalModel(
        data=df,
        treatment= treatment_name,  # Use the current treatment variable
        outcome= 'rating',

        cgm = (causal_digraph(
        correlation_dict=(webscrape_report()))))

        graph= cgm.dag
    

# Run analysis for this model
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        method_name=   "backdoor.econml.metalearners.XLearner",
        method_params={"init_params":{"models":"forest"}}
    )
    
    # Store the results
    model_results[treatment_name] = estimate
    
    # Optional: Print results for each treatment
    print(f"Treatment: {treatment_name}")
    print(estimate)
    print("-" * 50)