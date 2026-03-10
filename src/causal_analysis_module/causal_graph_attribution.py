"""
Causal Graph Attribution Module

This module provides graph-informed attribution analysis for causal inference.
It uses DoWhy and NetworkX to perform causal analysis based on a causal graph.
"""

import pandas as pd
import networkx as nx
from dowhy import CausalModel
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import numpy as np

def run_graph_informed_attribution(
    df: pd.DataFrame,
    causal_edges: List[Tuple[str, str]],
    treatment_col: Union[str, List[str]],
    outcome_col: str,
    method: str = "backdoor.linear_regression"
) -> Dict:
    """
    Run graph-informed attribution analysis.
    
    Args:
        df: DataFrame containing the data
        causal_edges: List of tuples representing causal edges (cause, effect)
        treatment_col: Name of the treatment column(s) - can be a string or list of strings
        outcome_col: Name of the outcome column
        method: Method for causal estimation (default: "backdoor.linear_regression")
        
    Returns:
        Dictionary containing attribution results. If multiple treatments provided,
        returns a dictionary with keys for each treatment.
    """
    # Create NetworkX graph from causal edges
    G = nx.DiGraph()
    G.add_edges_from(causal_edges)
    
    # Verify outcome column exists
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in DataFrame")
    
    # Handle single treatment vs list of treatments
    if isinstance(treatment_col, str):
        treatment_list = [treatment_col]
        single_treatment = True
    elif isinstance(treatment_col, list):
        treatment_list = treatment_col
        single_treatment = False
    else:
        raise TypeError(f"treatment_col must be a string or list of strings, got {type(treatment_col)}")
    
    # Verify all treatment columns exist
    missing_treatments = [t for t in treatment_list if t not in df.columns]
    if missing_treatments:
        raise ValueError(f"Treatment column(s) not found in DataFrame: {missing_treatments}")
    
    # Process each treatment
    all_results = {}
    
    for treatment in treatment_list:
        try:
            # Create DoWhy causal model
            model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome_col,
                graph=G
            )
            
            # Identify the causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Estimate the causal effect
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name=method
            )
            
            # Calculate naive attribution (simple correlation/regression)
            naive_effect = df[treatment].corr(df[outcome_col])
            
            # Prepare results for this treatment
            all_results[treatment] = {
                'treatment': treatment,
                'outcome': outcome_col,
                'causal_estimate': causal_estimate,
                'naive_effect': naive_effect,
                'identified_estimand': identified_estimand,
                'graph': G,
                'model': model
            }
            
        except Exception as e:
            print(f"Warning: Error analyzing treatment '{treatment}': {str(e)}")
            all_results[treatment] = {
                'treatment': treatment,
                'outcome': outcome_col,
                'error': str(e)
            }
            continue
    
    # Return single result if single treatment, otherwise return all results
    if single_treatment:
        return all_results[treatment_list[0]]
    else:
        return {
            'outcome': outcome_col,
            'treatments': all_results,
            'graph': G
        }


class CausalGraphAttribution:
    """
    Class for performing graph-informed causal attribution analysis.
    """
    
    def __init__(self, df: pd.DataFrame, outcome: str):
        """
        Initialize the CausalGraphAttribution analyzer.
        
        Args:
            df: DataFrame containing the data
            outcome: Name of the outcome variable
        """
        self.df = df.copy()
        self.outcome = outcome
        self.graph = None
        self.results = {}
        self.models = {}
        
        if outcome not in df.columns:
            raise ValueError(f"Outcome column '{outcome}' not found in DataFrame")
    
    def define_causal_graph_custom(self, edges: List[Tuple[str, str]]):
        """
        Define the causal graph using custom edges.
        
        Args:
            edges: List of tuples representing causal edges (cause, effect)
        """
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges)
        
        # Verify all nodes in edges exist in DataFrame
        all_nodes = set()
        for source, target in edges:
            all_nodes.add(source)
            all_nodes.add(target)
        
        missing_nodes = all_nodes - set(self.df.columns)
        if missing_nodes:
            raise ValueError(f"Nodes not found in DataFrame: {missing_nodes}")
    
    def visualize_causal_graph(self, filename: str = 'causal_graph.png', figsize: tuple = (12, 10)):
        """
        Visualize the causal graph.
        
        Args:
            filename: Output filename for the graph visualization
            figsize: Figure size tuple
        """
        if self.graph is None:
            raise ValueError("Causal graph not defined. Call define_causal_graph_custom() first.")
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Highlight outcome node
        node_colors = ['red' if node == self.outcome else 'lightblue' 
                      for node in self.graph.nodes()]
        
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=2000,
            font_size=10,
            font_weight='bold',
            with_labels=True,
            arrows=True,
            edge_color='gray',
            arrowsize=20,
            arrowstyle='->'
        )
        
        plt.title(f"Causal Graph (Outcome: {self.outcome})", size=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Causal graph saved to {filename}")
    
    def analyze_direct_effects(self, method: str = "backdoor.linear_regression"):
        """
        Analyze direct effects of all variables that directly affect the outcome.
        
        Args:
            method: Method for causal estimation
        """
        if self.graph is None:
            raise ValueError("Causal graph not defined. Call define_causal_graph_custom() first.")
        
        # Find all direct causes of the outcome
        direct_causes = [source for source, target in self.graph.edges() 
                        if target == self.outcome]
        
        results = {}
        
        for treatment in direct_causes:
            if treatment not in self.df.columns:
                print(f"Warning: Treatment '{treatment}' not found in DataFrame, skipping...")
                continue
            
            try:
                model = CausalModel(
                    data=self.df,
                    treatment=treatment,
                    outcome=self.outcome,
                    graph=self.graph
                )
                
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                causal_estimate = model.estimate_effect(
                    identified_estimand,
                    method_name=method
                )
                
                results[treatment] = {
                    'estimate': causal_estimate,
                    'estimand': identified_estimand
                }
                self.models[treatment] = model
                
            except Exception as e:
                print(f"Error analyzing {treatment}: {str(e)}")
                continue
        
        self.results['direct_effects'] = results
        return results
    
    def analyze_path_specific_effects(self, method: str = "backdoor.linear_regression"):
        """
        Analyze path-specific effects through intermediate variables.
        
        Args:
            method: Method for causal estimation
        """
        if self.graph is None:
            raise ValueError("Causal graph not defined. Call define_causal_graph_custom() first.")
        
        # Find all paths from treatments to outcome
        all_treatments = set([source for source, _ in self.graph.edges()])
        all_treatments = all_treatments - {self.outcome}
        
        path_results = {}
        
        for treatment in all_treatments:
            if treatment not in self.df.columns:
                continue
            
            # Find all paths from treatment to outcome
            try:
                paths = list(nx.all_simple_paths(self.graph, treatment, self.outcome))
                path_results[treatment] = {
                    'num_paths': len(paths),
                    'paths': paths
                }
            except:
                path_results[treatment] = {
                    'num_paths': 0,
                    'paths': []
                }
        
        self.results['path_specific_effects'] = path_results
        return path_results
    
    def compare_with_naive_attribution(self):
        """
        Compare graph-informed attribution with naive (correlation-based) attribution.
        """
        if 'direct_effects' not in self.results:
            raise ValueError("Run analyze_direct_effects() first.")
        
        comparison = {}
        
        for treatment, result in self.results['direct_effects'].items():
            # Get causal estimate value
            try:
                causal_value = result['estimate'].value
            except:
                causal_value = None
            
            # Calculate naive correlation
            naive_correlation = self.df[treatment].corr(self.df[self.outcome])
            
            comparison[treatment] = {
                'causal_effect': causal_value,
                'naive_correlation': naive_correlation,
                'difference': causal_value - naive_correlation if causal_value is not None else None
            }
        
        self.results['naive_comparison'] = comparison
        return comparison
    
    def export_results(self, filename: str = 'attribution_results.txt'):
        """
        Export results to a text file.
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Causal Graph Attribution Results\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Outcome Variable: {self.outcome}\n")
            f.write(f"Number of Nodes: {len(self.graph.nodes()) if self.graph else 0}\n")
            f.write(f"Number of Edges: {len(self.graph.edges()) if self.graph else 0}\n\n")
            
            if 'direct_effects' in self.results:
                f.write("Direct Effects:\n")
                f.write("-" * 80 + "\n")
                for treatment, result in self.results['direct_effects'].items():
                    f.write(f"\nTreatment: {treatment}\n")
                    try:
                        f.write(f"  Causal Estimate: {result['estimate'].value}\n")
                        f.write(f"  Estimate Details: {result['estimate']}\n")
                    except:
                        f.write(f"  Estimate: {result['estimate']}\n")
            
            if 'naive_comparison' in self.results:
                f.write("\n\nNaive Comparison:\n")
                f.write("-" * 80 + "\n")
                for treatment, comp in self.results['naive_comparison'].items():
                    f.write(f"\nTreatment: {treatment}\n")
                    f.write(f"  Causal Effect: {comp['causal_effect']}\n")
                    f.write(f"  Naive Correlation: {comp['naive_correlation']:.4f}\n")
                    if comp['difference'] is not None:
                        f.write(f"  Difference: {comp['difference']:.4f}\n")
        
        print(f"Results exported to {filename}")

