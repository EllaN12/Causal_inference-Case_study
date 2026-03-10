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
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
import logging, sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)



# 1. Construct the file URL:
file_path = os.path.abspath("Reports/data_report.html")
file_url = "file://" + file_path
print(file_url)

# Set up ChromeDriver with options:
executable_path = ChromeDriverManager().install()
service = Service(executable_path)
options = Options()


print(executable_path)
# Options to try to avoid detection 
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)



driver = webdriver.Chrome(service=service, options=options)
correlation_dict = {}

try:
    # 3. Open the local HTML file:
    driver.get(file_url)

    # Now you can use BeautifulSoup to parse the HTML:
    html = driver.page_source
    soup = soup(html, 'html.parser')
    
    alerts_table = soup.find('div', id='tab-pane-overview-alerts').find('table', class_='table-striped') 

    # ... your code to extract data from the soup ...
    print(alerts_table) 

    if alerts_table:
        for row in alerts_table.find_all('tr'):
            cells = row.find_all('td')
            if cells: # Check if there are cells in the ro
                link = cells[0].find('a')
                if link:
                    variable_name = link.text.strip()  # Extract the variable name
                    message = cells[0].text.replace(variable_name, "").replace(" has constant value '?'", "").replace(" is highly overall correlated with ", "").replace(" is highly imbalanced (", "").replace(" has ", "").replace(" missing values", "").replace(" is uniformly distributed", "").strip() # Extract the message
                badge = cells[1].find('span', class_='badge')
                alert_type = badge.text.strip() if badge else None # Extract the alert type
                other_fields_span = cells[0].find('span', attrs={'data-bs-toggle': 'tooltip'})
                other_fields = other_fields_span['data-bs-title'] if other_fields_span else None  # Extract other fields

                if alert_type == "High correlation":
                    correlation_dict[variable_name]= other_fields 

                
                print(f"Variable: {variable_name}")
                print(f"Message: {message}")
                print(f"Alert Type: {alert_type}")
                print(f"Other Fields: {other_fields}")
                print("-" * 20)
                
            
        
    else:
        print("Alerts table not found.")


finally:  # Ensure the driver quits even if there's an error
    

    driver.quit()  # Important: Close the browser


for variable_name, fields in correlation_dict.items():
    print(f"Variable: {variable_name}, Other Fields: {fields}")





print(correlation_dict)


#%%

# Parsing dictionarry relationships identified:
#%%
# Create a causal Grapgh
def extract_nodes_from_edges(edges: List[Tuple[str, str]]) -> Set[str]:
    """
    Extract unique nodes from list of edges
    """
    nodes = set()
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)
    return nodes


#%%
def convert_edges_to_networkx_format(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Convert edges to format expected by NetworkX
    """
    return [(source, target) for source, target in edges]


def convert_edges_to_networkx_format(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Convert edges to format expected by NetworkX
    """
    return [(source, target) for source, target in edges]


def parse_relationships_dict(correlation_dict: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Parse the relationships dictionary to create edges with error handling
    """
    additional_edges = []
    
    if not correlation_dict:
        print("Warning: Empty correlation dictionary provided")
        return additional_edges
        
    for source, targets in correlation_dict.items():
        if not targets:
            print(f"Warning: No targets found for source '{source}'")
            continue
            
        try:
            # Split targets string into individual targets and clean
            target_list = [t.strip() for t in targets.split(',') if t.strip()]
            
            # Create edges from source to each target
            for target in target_list:
                if target:  # Only add edge if target is not empty
                    additional_edges.append((str(source), str(target)))
                    
        except Exception as e:
            print(f"Error processing source '{source}': {str(e)}")
            continue
            
    print(f"Successfully created {len(additional_edges)} edges from correlation dictionary")
    return additional_edges

parse_relationships_dict(correlation_dict)

#%%
def create_causal_graph(correlation_dict: Dict[str, str]) -> CausalGraphicalModel:
    """
    Create causal graph incorporating both base causal relationships and correlations
    """
    try:
        # First, validate the correlation dictionary
        if not isinstance(correlation_dict, dict):
            raise ValueError("correlation_dict must be a dictionary")
            
        # Define base causal relationships
        base_edges = [
            # Direct Effects on Rating
            ('food_rating', 'rating'),
            ('service_rating', 'rating'),
            ('age_group', 'rating'),
            ('activity', 'rating'),
            ('lifestyle_cluster', 'rating'),
            ('personality', 'rating'),
            ('User_cuisine', 'rating'),
            
            # Demographic Effects
            ('age_group', 'budget'),
            ('age_group', 'User_cuisine'),
            ('age_group', 'drink_level'),
            ('age_group', 'dress_preference'),
            ('age_group', 'lifestyle_cluster'),
            ('activity', 'budget'),
            ('activity', 'transport'),
            ('activity', 'lifestyle_cluster'),
            ('marital_status', 'hijos'),
            ('marital_status', 'budget'),
            ('marital_status', 'user_ambience'),
            ('marital_status', 'lifestyle_cluster'),
            
            # Restaurant Selection Effects
            ('transport', 'accessibility'),
            ('transport', 'area'),
            ('budget', 'price'),
            ('lifestyle_cluster', 'Rcuisine_y'),
            
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
        
        # Parse and add edges from correlation dictionary
        correlation_edges = parse_relationships_dict(correlation_dict)
        
        # Create basic graph structure
        G = nx.DiGraph()
        
        # Add base edges
        G.add_edges_from(base_edges)
        logging.info(f"Added {len(base_edges)} base edges")
        
        # Add correlation edges
        if correlation_edges:
            G.add_edges_from(correlation_edges)
            logging.info(f"Added {len(correlation_edges)} correlation edges")
        
        # Extract all unique nodes and edges
        nodes = list(G.nodes())
        edges = list(G.edges())
        
        # Log edge details for debugging
        logging.debug("Edge details:")
        for edge in edges:
            logging.debug(f"  {edge[0]} -> {edge[1]}")
        
        # Create the causal graph
        cgm = CausalGraphicalModel(
            nodes=nodes,
            edges=edges
        )
        
        logging.info(f"Created causal graph with {len(nodes)} nodes and {len(edges)} edges")
        return cgm
        
    except Exception as e:
        logging.error(f"Error creating causal graph: {str(e)}")
        # Create fallback graph with base edges only
        G = nx.DiGraph()
        G.add_edges_from(base_edges)
        return CausalGraphicalModel(
            nodes=list(G.nodes()),
            edges=list(G.edges())
        )



causal_grapgh = create_causal_graph(correlation_dict)
#%%
# Create Causal Visualization.

def create_causal_visualization(cgm: CausalGraphicalModel, 
                              figsize: tuple = (20, 16)) -> plt.Figure:
    """
    Create a visualization of the causal graph with node categories
    """
    # Get the NetworkX graph
    G = cgm.dag
    
    # Define node categories and their colors
    node_categories = {
        'rating': ['rating'],
        'direct_effects': ['food_rating', 'service_rating', 'age_group', 'activity', 'lifestyle_cluster', 'personality', 'User_cuisine'],
        'demographics': [ 'marital_status'],
        'preferences': ['budget', 'drink_level', 'dress_preference'],
        'transport': ['transport', 'accessibility', 'area'],
        'personal': ['weight', 'height', 'color', 'hijos', 'smoker'],
        'restaurant': ['Rcuisine_y', 'price', 'user_ambience']
    }
    

    colors = {
        'rating': '#ff6666',        # Red
        'direct_effects': '#ffa366', # Orange
        'demographics': '#66b3ff',   # Blue
        'preferences': '#66ff66',    # Green
        'transport': '#ffff66',      # Yellow
        'personal': '#ff66ff',       # Pink
        'restaurant': '#a366ff'      # Purple
    }
    
    # Set up the plot
    plt.figure(figsize=figsize)
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes by category
    node_colors = []
    for node in G.nodes():
        # Find which category the node belongs to
        category = 'other'
        for cat, nodes in node_categories.items():
            if node in nodes:
                category = cat
                break
        node_colors.append(colors.get(category, '#gray'))
    
    # Draw the network
    nx.draw(G, pos,
            node_color=node_colors,
            node_size=2000,
            font_size=8,
            font_weight='bold',
            with_labels=True,
            arrows=True,
            edge_color='gray',
            arrowsize=20,
            arrowstyle='->',
            node_shape='o')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], 
                                 marker='o', 
                                 color='w',
                                 markerfacecolor=color,
                                 markersize=10,
                                 label=category.replace('_', ' ').title())
                      for category, color in colors.items()]
    
    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              title='Node Categories',
              title_fontsize=12,
              fontsize=10)
    
    # Add title
    plt.title("Restaurant Rating Causal Graph", pad=20, size=16)
    
    # Remove axes
    plt.axis('off')
    
    # Add graph statistics as text
    stats_text = f"Nodes: {len(G.nodes())}\n"
    stats_text += f"Edges: {len(G.edges())}\n"
    stats_text += f"Average degree: {sum(dict(G.degree()).values())/len(G):0.2f}"
    
    plt.text(0.95, 0.05, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='bottom',
            horizontalalignment='right')
    
    plt.tight_layout()
    return plt


create_causal_visualization(causal_grapgh)


#%%
def verify_graph_structure(cgm: CausalGraphicalModel) -> Dict:
    """
    Verify the structure of the created causal graph
    """
    try:
        G = cgm.dag
        verification = {
            'total_nodes': len(G.nodes()),
            'total_edges': len(G.edges()),
            'has_rating_node': 'rating' in G.nodes(),
            'direct_effects_to_rating': list(G.predecessors('rating')),
            'isolated_nodes': list(nx.isolates(G)),
            'is_dag': nx.is_directed_acyclic_graph(G)
        }
        logging.info("Graph structure verification completed")
        return verification
        
    except Exception as e:
        logging.error(f"Error verifying graph structure: {str(e)}")
        return {'error': str(e)}

print (verify_graph_structure(
       cgm = create_causal_graph(correlation_dict)))
#%%

""""def analyze_causal_relationships(cgm):
    """
    Analyze key aspects of the causal graph
    """
    # Get all nodes
    nodes = cgm.dag.nodes()
    
    # Analyze direct causes of rating
    direct_causes_rating = [edge[0] for edge in cgm.dag.in_edges('rating')]
    
    # Analyze nodes by their role
    root_nodes = [node for node in nodes if cgm.dag.in_degree(node) == 0]
    leaf_nodes = [node for node in nodes if cgm.dag.out_degree(node) == 0]
    intermediary_nodes = [node for node in nodes 
                         if node not in root_nodes and node not in leaf_nodes]
    
    analysis = {
        'total_nodes': len(nodes),
        'total_edges': len(cgm.dag.edges()),
        'direct_causes_of_rating': direct_causes_rating,
        'root_nodes': root_nodes,
        'leaf_nodes': leaf_nodes,
        'intermediary_nodes': intermediary_nodes
    }
    
    return analysis """"


def visualize_causal_graph(cgm):
    """
    Create a visualization of the causal graph with custom styling
    """
    plt.figure(figsize=(15, 10))
    
    # Convert to NetworkX graph for visualization
    G = cgm.dag
    
    # Define node categories for coloring
    node_categories = {
        'demographic': ['age_group', 'marital_status', 'activity', 'lifestyle_cluster'],
        'preference': ['budget', 'User_cuisine', 'drink_level', 'dress_preference', 'transport'],
        'rating': ['rating', 'food_rating', 'service_rating'],
        'personal': ['weight', 'height', 'personality', 'color', 'hijos', 'smoker'],
        'restaurant': ['accessibility', 'area', 'price', 'Rcuisine_y']
    }
    
    # Create color mapping
    node_colors = {}
    color_map = {
        'demographic': '#e6f3ff',
        'preference': '#e6ffe6',
        'rating': '#ffe6e6',
        'personal': '#f0e6ff',
        'restaurant': '#fff0e6'
    }
    
    for category, nodes in node_categories.items():
        for node in nodes:
            node_colors[node] = color_map[category]
    
    # Set default color for any uncategorized nodes
    node_color_list = [node_colors.get(node, '#ffffff') for node in G.nodes()]
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the graph
    nx.draw(G, pos,
            node_color=node_color_list,
            with_labels=True,
            node_size=2000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20)
    
    plt.title("Restaurant Rating Causal Graph")
    plt.axis('off')
    return plt


def identify_key_pathways(cgm):
    """
    Identify and analyze key causal pathways to rating
    """
    G = cgm.dag
    
    # Find all paths to rating
    paths_to_rating = {}
    for node in G.nodes():
        if node != 'rating':
            try:
                paths = list(nx.all_simple_paths(G, node, 'rating'))
                if paths:
                    paths_to_rating[node] = paths
            except nx.NetworkXNoPath:
                continue
    
    return paths_to_rating

# Example usage
if __name__ == "__main__":
    # Create the causal graph
    cgm = create_causal_graph()
    
    # Analyze the graph
    analysis = analyze_causal_relationships(cgm)
    print("\nCausal Graph Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    # Identify key pathways
    pathways = identify_key_pathways(cgm)
    print("\nKey Pathways to Rating:")
    for node, paths in pathways.items():
        print(f"\nFrom {node}:")
        for path in paths:
            print(f"  {' -> '.join(path)}")
    
    # Visualize the graph
    plt = visualize_causal_graph(cgm)
    plt.show()




<table class="table table-striped"><tbody><tr><td><a href="#pp_var_1958749664926579591"><code>Business_hours</code></a> is highly overall correlated with <code>area</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="area, state, zip">2 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_9121400791270513098"><code>accessibility</code></a> is highly overall correlated with <code>area</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="area, city, placeID, restaurant_specialty, smoking_area, state, zip">6 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_6505678126749485994"><code>activity</code></a> is highly overall correlated with <code>age_group</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="age_group, marital_status, p.longitude, weight">3 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-9101062890890038943"><code>age</code></a> is highly overall correlated with <code>age_group</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="age_group, ambience, birth_year, drink_level, hijos">4 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-5554233853932471152"><code>age_group</code></a> is highly overall correlated with <code>activity</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="activity, age, birth_year, color, p.longitude, patrons_restaurant_distance">5 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_8423411984832590710"><code>alcohol</code></a> is highly overall correlated with <code>restaurant_specialty</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="restaurant_specialty, url, zip">2 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_1780827768041950105"><code>ambiance</code></a> is highly overall correlated with <code>restaurant_specialty</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="restaurant_specialty, state">1 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_2149280729532653732"><code>ambience</code></a> is highly overall correlated with <code>age</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="age, birth_year, color, height, interest, transport, weight">6 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-8988845978301601035"><code>area</code></a> is highly overall correlated with <code>Business_hours</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="Business_hours, accessibility, smoking_area, state">3 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_6679910995680182724"><code>birth_year</code></a> is highly overall correlated with <code>age</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="age, age_group, ambience, drink_level, hijos">4 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_293362163062244439"><code>budget</code></a> is highly overall correlated with <code>color</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="color, height, hijos, transport">3 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_6236043459790324177"><code>city</code></a> is highly overall correlated with <code>accessibility</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="accessibility, country, latitude, location_cluster, longitude, other_services, p.latitude, p.longitude, placeID, state, zip">10 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_6687643619691513724"><code>color</code></a> is highly overall correlated with <code>age_group</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="age_group, ambience, budget, drink_level, food_rating, hijos, interest, personality, rating, service_rating, transport">10 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-3185681048921875716"><code>country</code></a> is highly overall correlated with <code>city</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="city, restaurant_specialty, state, zip">3 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-1650288806968959149"><code>dress_code</code></a> is highly overall correlated with <code>url</code></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_6822277003026177529"><code>drink_level</code></a> is highly overall correlated with <code>age</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="age, birth_year, color, height, transport, weight">5 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_2551324760613085485"><code>food_rating</code></a> is highly overall correlated with <code>color</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="color, height, hijos, personality, rating, service_rating, weight">6 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_3620307622254911381"><code>franchise</code></a> is highly overall correlated with <code>location_cluster</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="location_cluster, restaurant_specialty, zip">2 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-6939554151446200194"><code>height</code></a> is highly overall correlated with <code>ambience</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="ambience, budget, drink_level, food_rating, interest, personality, rating, transport, weight">8 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_884945857066056626"><code>hijos</code></a> is highly overall correlated with <code>age</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="age, birth_year, budget, color, food_rating, rating, transport">6 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_1799465583985044801"><code>interest</code></a> is highly overall correlated with <code>ambience</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="ambience, color, height, rating">3 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_3121070849282128254"><code>latitude</code></a> is highly overall correlated with <code>city</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="city, location_cluster, state, zip">3 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-8684456203240582530"><code>location_cluster</code></a> is highly overall correlated with <code>city</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="city, franchise, latitude, longitude, p.latitude, p.longitude, restaurant_specialty, state, zip">8 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_2322278189700707437"><code>longitude</code></a> is highly overall correlated with <code>city</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="city, location_cluster, state, zip">3 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-5122842302050025555"><code>marital_status</code></a> is highly overall correlated with <code>activity</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="activity, transport">1 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_4075237506425857646"><code>other_services</code></a> is highly overall correlated with <code>city</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="city, restaurant_specialty, state">2 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_6650865745658438830"><code>p.latitude</code></a> is highly overall correlated with <code>city</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="city, location_cluster, state">2 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_6978851293643007702"><code>p.longitude</code></a> is highly overall correlated with <code>activity</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="activity, age_group, city, location_cluster, patrons_restaurant_distance, state">5 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-351578062456421512"><code>parking_lot</code></a> is highly overall correlated with <code>restaurant_specialty</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="restaurant_specialty, zip">1 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_5248172779119302217"><code>patrons_restaurant_distance</code></a> is highly overall correlated with <code>age_group</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="age_group, p.longitude">1 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_1345154570155919642"><code>personality</code></a> is highly overall correlated with <code>color</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="color, food_rating, height, rating, weight">4 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_2245129410092678378"><code>placeID</code></a> is highly overall correlated with <code>accessibility</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="accessibility, city, restaurant_specialty, state, zip">4 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-2095598002233567461"><code>price</code></a> is highly overall correlated with <code>restaurant_specialty</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="restaurant_specialty, url, zip">2 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-2114249187477870707"><code>rating</code></a> is highly overall correlated with <code>color</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="color, food_rating, height, hijos, interest, personality, service_rating">6 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_5552623071763888703"><code>restaurant_specialty</code></a> is highly overall correlated with <code>accessibility</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="accessibility, alcohol, ambiance, country, franchise, location_cluster, other_services, parking_lot, placeID, price, smoking_area">10 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-1881134645743709469"><code>service_rating</code></a> is highly overall correlated with <code>color</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="color, food_rating, rating">2 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-3986505599822090425"><code>smoking_area</code></a> is highly overall correlated with <code>accessibility</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="accessibility, area, restaurant_specialty">2 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_7791717604239632581"><code>state</code></a> is highly overall correlated with <code>Business_hours</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="Business_hours, accessibility, ambiance, area, city, country, latitude, location_cluster, longitude, other_services, p.latitude, p.longitude, placeID, zip">13 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-77453177052106321"><code>transport</code></a> is highly overall correlated with <code>ambience</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="ambience, budget, color, drink_level, height, hijos, marital_status, weight">7 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_7344025960226245411"><code>url</code></a> is highly overall correlated with <code>alcohol</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="alcohol, dress_code, price, zip">3 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-3026241257979484893"><code>weight</code></a> is highly overall correlated with <code>activity</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="activity, ambience, drink_level, food_rating, height, personality, transport">6 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_-3227584300337695938"><code>zip</code></a> is highly overall correlated with <code>Business_hours</code> and <span data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="Business_hours, accessibility, alcohol, city, country, franchise, latitude, location_cluster, longitude, parking_lot, placeID, price, state, url">13 other fields</span></td><td><span class="badge text-bg-secondary">High correlation</span></td></tr><tr><td><a href="#pp_var_4667551590583924293"><code>smoker</code></a> is highly imbalanced (65.6%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_-5122842302050025555"><code>marital_status</code></a> is highly imbalanced (84.2%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_370321761805882128"><code>religion</code></a> is highly imbalanced (71.0%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_6505678126749485994"><code>activity</code></a> is highly imbalanced (73.6%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_4875916516666032629"><code>Upayment</code></a> is highly imbalanced (56.2%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_6236043459790324177"><code>city</code></a> is highly imbalanced (70.9%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_7791717604239632581"><code>state</code></a> is highly imbalanced (56.7%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_-3185681048921875716"><code>country</code></a> is highly imbalanced (61.2%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_-1650288806968959149"><code>dress_code</code></a> is highly imbalanced (69.3%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_7344025960226245411"><code>url</code></a> is highly imbalanced (67.2%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_1780827768041950105"><code>ambiance</code></a> is highly imbalanced (85.1%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_-8988845978301601035"><code>area</code></a> is highly imbalanced (59.0%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_4075237506425857646"><code>other_services</code></a> is highly imbalanced (84.3%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_5919969639934050706"><code>cuisine_match_score</code></a> is highly imbalanced (53.5%) </td><td><span class="badge text-bg-primary">Imbalance</span></td></tr><tr><td><a href="#pp_var_-1912143054417428616"><code>start_time</code></a> is an unsupported type, check if it needs cleaning or further analysis </td><td><span class="badge text-bg-warning">Unsupported</span></td></tr><tr><td><a href="#pp_var_-3207340698141660022"><code>end_time</code></a> is an unsupported type, check if it needs cleaning or further analysis </td><td><span class="badge text-bg-warning">Unsupported</span></td></tr></tbody></table>

