"""
COMPLETE CAUSAL GRAPH MODULE - TEST CASE 2
===========================================
Integrates:
1. Pre-defined causal edges (correlation_edges)
2. Graph construction (define_causal_graph_custom)
3. Graph visualization (visualize_causal_graph)
4. Attribution analysis foundation

Ready for causal inference and attribution modeling.
"""
#%%
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import os


#CAUSAL EDGES: defined from the correlation analysis 

correlation_edges = [
    
    # Outcome (7 edges - direct effects ON rating)
    ("color", "rating"),           
    ("height", "rating"),           
    ("hijos", "rating"),           
    ("interest", "rating"),        
    ("personality", "rating"),
    ("food_rating", "rating"),     
    ("service_rating", "rating"),   
    
    # Business_hours (1 edge)
    ("area", "Business_hours"),
    
    # Food rating (2 edges)
    ("height", "food_rating"),      
    ("hijos", "food_rating"),      
    
    # Service rating (1 edge)
    ("color", "service_rating"),    
    
    # height (6 edges)
    ("height", "ambience"),
    ("height", "budget"),
    ("height", "drink_level"),      
    ("height", "personality"),
    ("height", "transport"),
    ("height", "weight"),
    
    # Interest (3 edges)
    ("interest", "ambience"),
    ("interest", "color"),
    ("interest", "height"),
    
    # Color edges (7 edges)
    ("age_group", "color"),
    ("ambience", "color"),
    ("budget", "color"),
    ("drink_level", "color"),       
    ("hijos", "color"),
    ("personality", "color"),       
    
    # Activity (3 edges)
    ("marital_status", "activity"), 
    ("weight", "activity"),
    ("age_group", "activity"),
    
    # age_group (1 edge)
    ("age_group", "patrons_restaurant_distance"),
    
    # accessibility (3 edges)
    ("accessibility", "area"),
    ("accessibility", "restaurant_specialty"),  
    ("accessibility", "smoking_area"),          
    
    # Children / hijos (2 edges)
    ("hijos", "budget"),
    ("hijos", "transport"),
    
    # Personality (4 edges)
    ("personality", "color"),           
    ("personality", "food_rating"),     
    ("height", "personality"),  
    ("weight", "personality"),
    
    # alcohol (2 edges)
    ("alcohol", "restaurant_specialty"),    
    ("alcohol", "url"),
    
    # ambiance (1 edge)
    ("ambiance", "restaurant_specialty"),   
    
    # ambience (5 edges)
    ("ambience", "color"),
    ("ambience", "height"),
    ("ambience", "interest"),
    ("ambience", "transport"),
    ("ambience", "weight"),
    
    # area (3 edges)
    ("area", "accessibility"),
    ("area", "smoking_area"),               
    ("area", "Business_hours"),
    
    # budget (3 edges)
    ("budget", "color"),
    ("budget", "height"),
    ("budget", "transport"),
    
    # drink_level (3 edges)
    ("drink_level", "height"),             
    ("drink_level", "transport"),          
    ("drink_level", "weight"),              
    
    # franchise (3 edges)
    ("franchise", "location_cluster"),      
    ("franchise", "restaurant_specialty"),  
    ("franchise", "zip"),
    
    # price (3 edges)
    ("restaurant_specialty", "price"),      
    ("url", "price"),                  
    ("zip", "price"),            
    
    # location_cluster (3 edges)
    ("location_cluster", "franchise"),          
    ("location_cluster", "restaurant_specialty"), 
    ("zip", "location_cluster"),                 
    
    # marital_status (2 edges)
    ("marital_status", "activity"),         
    ("marital_status", "transport"),        
    
    # other_services (1 edge)
    ("other_services", "restaurant_specialty"),  
    
    # parking_lot (2 edges)
    ("restaurant_specialty", "parking_lot"),     
    ("zip", "parking_lot"),                      
    
    # patrons_restaurant_distance (1 edge)
    ("patrons_restaurant_distance", "age_group"),
    
    # restaurant_specialty (8 edges)
    ("restaurant_specialty", "accessibility"),  
    ("restaurant_specialty", "alcohol"),
    ("restaurant_specialty", "ambiance"),
    ("restaurant_specialty", "location_cluster"),  
    ("restaurant_specialty", "other_services"),    
    ("restaurant_specialty", "parking_lot"),       
    ("restaurant_specialty", "price"),
    ("restaurant_specialty", "smoking_area"),      
    
    # transport (5 edges)
    ("transport", "ambience"),
    ("transport", "color"),
    ("transport", "drink_level"),      
    ("marital_status", "transport"),    
    ("budget", "transport"),
    
    # weight (6 edges)
    ("weight", "activity"),
    ("weight", "ambience"),
    ("weight", "drink_level"),          
    ("weight", "food_rating"),          
    ("weight", "personality"),
    ("weight", "transport")
]


# Causal graph construction function
#%%
def define_causal_graph_custom(edges, outcome='rating', visualize=True, 
                              save_path=None):
    """
    Define custom causal graph from edge list with integrated visualization.
    
    This function:
    1. Creates a directed graph from your causal edges
    2. Identifies treatment variables (direct factors that affect the outcome)
    3. visualizes the causal graph
    4. Returns graph object ready for attribution analysis
    
    Parameters:
    -----------
    edges : list of tuples
        List of (cause, effect) pairs defining causal relationships
        Example: [('food_rating', 'rating'), ('service_rating', 'rating')]
    outcome : str, optional
        The outcome variable name (default: 'rating')
    visualize : bool, optional
        Whether to create visualization (default: True)
    save_path : str, optional
        Path to save visualization (default: 'causal_graph.png')
    
    Returns:
    --------
    graph : nx.DiGraph
        The created directed graph with all causal relationships
    treatments : list
        List of nodes that directly affect the outcome (treatment variables)
    
    Example:
    --------
    >>> # Using pre-defined edges
    >>> graph, treatments = define_causal_graph_custom(
    ...     correlation_edges, 
    ...     outcome='rating',
    ...     visualize=True
    ... )
    >>> print(f"Treatments: {treatments}")
    >>> # Use graph for attribution analysis
    """
    
    print("\n" + "="*80)
    print("CAUSAL GRAPH CONSTRUCTION - TEST CASE 2")
    print("="*80)
    
    # Create directed graph
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    
    print(f"\n✓ Causal graph created:")
    print(f"  • Nodes: {graph.number_of_nodes()}")
    print(f"  • Edges: {graph.number_of_edges()}")
    print(f"  • Outcome variable: '{outcome}'")
    
    # Extract treatments (nodes with direct edges to outcome)
    if outcome in graph.nodes():
        treatments = list(graph.predecessors(outcome))
        treatments_sorted = sorted(treatments)
        
        print(f"\n✓ Identified {len(treatments)} direct treatment variables:")
        print(f"  (Variables with direct causal effect on '{outcome}')")
        for i, treatment in enumerate(treatments_sorted, 1):
            print(f"  {i}. {treatment} → {outcome}")
    else:
        treatments = []
        print(f"\n⚠ Warning: Outcome '{outcome}' not found in graph nodes")
        print(f"  Available nodes: {sorted(graph.nodes())[:10]}... (showing first 10)")
    
    # Graph statistics
    print(f"\n📊 Graph Statistics:")
    print(f"  • Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")
    print(f"  • Is DAG (Directed Acyclic Graph): {nx.is_directed_acyclic_graph(graph)}")
    
    if not nx.is_directed_acyclic_graph(graph):
        print(f"  ⚠ Warning: Graph contains cycles!")
        try:
            cycles = list(nx.simple_cycles(graph))
            print(f"  • Number of cycles: {len(cycles)}")
            if cycles:
                print(f"  • Example cycle: {cycles[0]}")
        except:
            pass
    
    # Visualize if requested
    if visualize:
        print(f"\n📊 Creating causal graph visualization...")
        fig = visualize_causal_graph(
            graph=graph, 
            treatments=treatments, 
            outcome=outcome, 
            save_path=save_path
        )
        print(f"✓ Visualization complete!")
    
    print("\n" + "="*80)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("="*80)
    print(f"\n✓ Ready for:")
    print(f"  • Attribution analysis (which treatments matter most?)")
    print(f"  • Mediation analysis (indirect effects through other variables)")
    print(f"  • Path analysis (specific causal pathways)")
    print(f"  • Confounder identification (backdoor paths)")
    
    return graph, treatments

#%%
# Causal graph visualization function
def visualize_causal_graph(graph, treatments=None, outcome='rating', 
                          save_path=None):
    """
    Visualize causal graph with color-coded nodes and clear legend.
    
    Color Scheme:
    - RED: Outcome variable
    - GOLD: Direct treatment variables (direct factors that affect the outcome)
    - GREEN: User-related variables
    - MAGENTA: Restaurant-related variables
    - GRAY: Other variables (not directly related to the outcome nor treatment)
    
    Parameters:
    -----------
    graph : nx.DiGraph
        The causal graph to visualize
    treatments : list, optional
        List of treatment nodes to highlight (default: None)
    outcome : str, optional
        The outcome variable name (default: 'rating')
    save_path : str, optional
        Path to save the visualization (default: 'causal_graph.png')
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    if graph is None:
        print("⚠ No causal graph provided.")
        return None
    
    # Set default save path if not provided
    if save_path is None:
        save_path = os.path.abspath(os.path.join('Reports', 'causal_graph.png'))
    
    # Ensure save_path is a string (not None)
    save_path = str(save_path)
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(save_path)
    if dir_path:  # Only create directory if path contains a directory component
        os.makedirs(dir_path, exist_ok=True)

    # Define node categories
    node_categories = {
        'Outcome': [outcome],
        'User_Demographics_Attributes_Location_Preferences': [
            'age_group', 'marital_status', 'hijos', 'weight', 'height', 'color',
            'activity', 'ambience', 'interest', 'personality', 'drink_level', 
            'budget', 'transport', 'location_cluster', 'patrons_restaurant_distance'
        ],
        'Restaurant_Location_Features_Attributes': [
            'area', 'zip', 'restaurant_specialty', 'franchise', 'price', 
            'Business_hours', 'url', 'dress_code', 'accessibility', 'parking_lot', 
            'smoking_area', 'alcohol', 'ambiance', 'other_services'
        ],
        'Other': []
    }

    category_colors = {
        'Outcome': '#FF0000',              # Red
        'User_Demographics_Attributes_Location_Preferences': '#00FF00',  # Green
        'Restaurant_Location_Features_Attributes': '#FF00FF',  # Magenta
        'Other': '#CCCCCC'  # Gray
    }

    def get_node_category(node):
        """Get the category for a given node."""
        for category, nodes in node_categories.items():
            if node in nodes:
                return category
        return 'Other'

    # Color mapping
    node_colors = []
    node_to_category = {}
    for node in graph.nodes():
        if treatments and node in treatments:
            node_colors.append('#FFD700')  # Gold for treatments
            node_to_category[node] = 'Treatment'
        else:
            category = get_node_category(node)
            node_to_category[node] = category
            node_colors.append(category_colors.get(category, category_colors['Other']))

    # Create figure with extra space for legend
    fig, ax = plt.subplots(figsize=(26, 18))

    # Layout with good spacing
    pos = nx.spring_layout(graph, seed=42, k=4, iterations=100)

    # Draw edges with BLACK arrows (high visibility)
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='#000000',
        arrows=True,
        arrowsize=35,
        arrowstyle='->',
        width=3.5,
        alpha=0.5,
        ax=ax,
        node_size=1800,
        connectionstyle='arc3,rad=0.2'
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=1800,
        alpha=0.95,
        ax=ax,
        edgecolors='black',
        linewidths=3
    )

    # Draw labels with white background for readability
    for node, (x, y) in pos.items():
        ax.text(x, y, node,
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='white',
                         edgecolor='black',
                         alpha=0.9,
                         linewidth=1.5),
                zorder=3)

    # Create legend elements
    legend_elements = []
    
    # Add core categories
    legend_elements.append(
        Patch(facecolor='#FF0000', edgecolor='black', linewidth=2, label='Outcome')
    )
    
    legend_elements.append(
        Patch(facecolor='#00FF00', edgecolor='black', linewidth=2, label='User Variables')
    )
    
    legend_elements.append(
        Patch(facecolor='#FF00FF', edgecolor='black', linewidth=2, label='Restaurant Variables')
    )
    
    # Add treatments if they exist
    if treatments and len(treatments) > 0:
        legend_elements.append(
            Patch(facecolor='#FFD700', edgecolor='black', linewidth=2, label='Direct Factors')
        )
    
    # Add Other if any nodes are categorized as Other
    other_nodes = [n for n, cat in node_to_category.items() if cat == 'Other']
    if other_nodes:
        legend_elements.append(
            Patch(facecolor='#CCCCCC', edgecolor='black', linewidth=2, label='Other')
        )
    
    # Position legend OUTSIDE the plot area
    legend = ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),  # Places legend outside on right
        fontsize=16,
        frameon=True,
        fancybox=True,
        shadow=True,
        title='Categories',
        title_fontsize=18
    )
    
    # Set legend frame properties
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(2)

    ax.set_title('Your Causal Graph for Attribution Analysis',
                 fontsize=22, fontweight='bold', pad=25)
    ax.axis('off')
    ax.margins(0.1)

    # Ensure save_path is set (safety check)
    if save_path is None:
        save_path = os.path.abspath(os.path.join('Reports', 'causal_graph.png'))
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    # Ensure save_path is a string (not None)
    save_path = str(save_path)
    
    # Save with tight bounding box to include legend
    plt.tight_layout()
    plt.savefig(save_path, dpi=350, bbox_inches='tight', facecolor='white')
    
    print(f"\n📊 Causal graph visualization saved to '{save_path}'")
    print("✅ Legend positioned OUTSIDE the graph on the right side")

    # Print categorization report
    print("\n📋 Node Categories in Graph:")
    
    category_counts = {
        'Outcome': 0,
        'User_Demographics_Attributes_Location_Preferences': 0,
        'Restaurant_Location_Features_Attributes': 0,
        'Treatment': 0,
        'Other': 0
    }
    
    for node, category in node_to_category.items():
        if category in category_counts:
            category_counts[category] += 1
    
    print(f"  • Outcome: {category_counts['Outcome']} node(s)")
    print(f"  • User Variables: {category_counts['User_Demographics_Attributes_Location_Preferences']} node(s)")
    print(f"  • Restaurant Variables: {category_counts['Restaurant_Location_Features_Attributes']} node(s)")
    
    if treatments:
        print(f"  • Direct Factors (treatments): {category_counts['Treatment']} node(s)")
        print(f"    Treatments: {', '.join(sorted(treatments))}")
    
    if category_counts['Other'] > 0:
        print(f"  ⚠️  Other (uncategorized): {category_counts['Other']} node(s)")
        other_list = [n for n, c in node_to_category.items() if c == 'Other']
        print(f"      {other_list}")

    return fig



# Functions to support causal analysis
#%%

# Get all causal paths from source to target

def get_causal_paths(graph, source, target, max_length=5):
    """
    Get all causal paths from source to target.
    
    Useful for understanding HOW one variable affects another.
    
    Parameters:
    -----------
    graph : nx.DiGraph
        The causal graph
    source : str
        Starting variable
    target : str
        Ending variable
    max_length : int, optional
        Maximum path length to consider (default: 5)
    
    Returns:
    --------
    paths : list of lists
        All paths from source to target
    
    Example:
    --------
    >>> paths = get_causal_paths(graph, 'age_group', 'rating')
    >>> for path in paths:
    ...     print(" → ".join(path))
    """
    try:
        paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
        return paths
    except nx.NetworkXNoPath:
        return []






#%%
def get_mediators(graph, treatment, outcome):
    """
    Get variables that mediate the effect of treatment on outcome.
    
    A mediator is on a path from treatment to outcome.
    
    Parameters:
    -----------
    graph : nx.DiGraph
        The causal graph
    treatment : str
        Treatment variable
    outcome : str
        Outcome variable
    
    Returns:
    --------
    mediators : set
        Set of mediating variables
    
    Example:
    --------
    >>> mediators = get_mediators(graph, 'age_group', 'rating')
    >>> print(f"Mediators: {mediators}")
    """
    paths = get_causal_paths(graph, treatment, outcome)
    mediators = set()
    
    for path in paths:
        # Mediators are variables between treatment and outcome
        if len(path) > 2:
            mediators.update(path[1:-1])
    
    return mediators


#%%

def get_confounders(graph, treatment, outcome):
    """
    Get potential confounders of treatment-outcome relationship.
    
    A confounder affects both treatment and outcome (backdoor path).
    
    Parameters:
    -----------
    graph : nx.DiGraph
        The causal graph
    treatment : str
        Treatment variable
    outcome : str
        Outcome variable
    
    Returns:
    --------
    confounders : set
        Set of potential confounders
    
    Example:
    --------
    >>> confounders = get_confounders(graph, 'food_rating', 'rating')
    >>> print(f"Confounders: {confounders}")
    """
    # Simplified: variables that are predecessors of both treatment and outcome
    treatment_ancestors = set(graph.predecessors(treatment))
    outcome_ancestors = set(graph.predecessors(outcome))
    
    confounders = treatment_ancestors.intersection(outcome_ancestors)
    
    return confounders

#%%
def print_graph_summary(graph, outcome='rating'):
    """
    Print comprehensive summary of causal graph structure.
    
    Parameters:
    -----------
    graph : nx.DiGraph
        The causal graph
    outcome : str
        Outcome variable name
    """
    print("\n" + "="*80)
    print("CAUSAL GRAPH SUMMARY")
    print("="*80)
    
    print(f"\n📊 Basic Statistics:")
    print(f"  • Total variables: {graph.number_of_nodes()}")
    print(f"  • Total relationships: {graph.number_of_edges()}")
    print(f"  • Outcome variable: {outcome}")
    
    # Direct causes of outcome
    if outcome in graph.nodes():
        direct_causes = list(graph.predecessors(outcome))
        print(f"\n🎯 Direct Causes of {outcome} ({len(direct_causes)}):")
        for cause in sorted(direct_causes):
            print(f"  • {cause}")
        
        # Effects of outcome (shouldn't be many if outcome is truly outcome)
        direct_effects = list(graph.successors(outcome))
        if direct_effects:
            print(f"\n⚠ Variables affected by {outcome} ({len(direct_effects)}):")
            for effect in sorted(direct_effects):
                print(f"  • {effect}")
            print(f"  Note: If {outcome} is truly the outcome, it shouldn't cause other variables")
    
    # Most influential variables (high out-degree)
    out_degrees = dict(graph.out_degree())
    top_influencers = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n🌟 Most Influential Variables (by out-degree):")
    for var, degree in top_influencers:
        if degree > 0:
            print(f"  • {var}: affects {degree} other variable(s)")
    
    # Most influenced variables (high in-degree)
    in_degrees = dict(graph.in_degree())
    top_influenced = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n📍 Most Influenced Variables (by in-degree):")
    for var, degree in top_influenced:
        if degree > 0:
            print(f"  • {var}: affected by {degree} other variable(s)")
    
    print("\n" + "="*80)


# ============================================================================
# STEP 5: MAIN EXECUTION FUNCTION
# ============================================================================
#%%
def run_complete_causal_analysis(edges=None, outcome='rating', 
                                save_path=None):
    """
    Run complete causal graph analysis: construct + visualize + summarize.
    
    This is your one-stop function for TEST CASE 2.
    
    Parameters:
    -----------
    edges : list of tuples, optional
        Causal edges. If None, uses pre-defined correlation_edges
    outcome : str, optional
        Outcome variable (default: 'rating')
    save_path : str, optional
        Path to save visualization (default: 'causal_graph.png')
    
    Returns:
    --------
    graph : nx.DiGraph
        The causal graph
    treatments : list
        Direct treatment variables
    
    Example:
    --------
    >>> # Use pre-defined edges
    >>> graph, treatments = run_complete_causal_analysis()
    >>> 
    >>> # Or use custom edges
    >>> my_edges = [('A', 'B'), ('B', 'C'), ('C', 'rating')]
    >>> graph, treatments = run_complete_causal_analysis(my_edges)
    """
    
    # Use pre-defined edges if none provided
    if edges is None:
        edges = correlation_edges
        print("\n✓ Using pre-defined correlation_edges")
    
    # Step 1: Construct graph
    graph, treatments = define_causal_graph_custom(
        edges=edges,
        outcome=outcome,
        visualize=True,
        save_path=save_path
    )
    
    # Step 2: Print summary
    print_graph_summary(graph, outcome=outcome)
    
    # Step 3: Analyze key relationships
    print("\n" + "="*80)
    print("KEY CAUSAL RELATIONSHIPS")
    print("="*80)
    
    for treatment in sorted(treatments)[:5]:  # Show first 5 treatments
        print(f"\n--- {treatment} → {outcome} ---")
        
        # Check for mediation
        mediators = get_mediators(graph, treatment, outcome)
        if mediators:
            print(f"  Mediators: {sorted(mediators)}")
        else:
            print(f"  No mediators (direct effect only)")
        
        # Check for confounders
        confounders = get_confounders(graph, treatment, outcome)
        if confounders:
            print(f"  Potential confounders: {sorted(confounders)}")
        else:
            print(f"  No common causes identified")
    
    print("\n" + "="*80)
    print("✅ CAUSAL ANALYSIS COMPLETE - READY FOR ATTRIBUTION")
    print("="*80)
    
    return graph, treatments


#

# %%
graph, treatments = run_complete_causal_analysis()