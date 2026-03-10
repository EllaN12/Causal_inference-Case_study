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
import io
import tempfile
from causal_analysis_module.analysis import data_preprocessing



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

def create_and_visualize_graph(graph_spec, plot=True):
    """
    Creates a directed acyclic graph (DAG) from a graphviz specification and visualizes it.

    Args:
        graph_spec: A string containing the graphviz graph specification.
        plot: If True, plots the graph using matplotlib (requires graphviz installation).

    Returns:
        A networkx.DiGraph object representing the DAG.
    """

    dot_data = io.StringIO(graph_spec)
    dot = Digraph()  # Create a Digraph object
    dot.source = graph_spec # assign the graph spec to the source attribute
    graph = nx.DiGraph(nx.drawing.nx_agraph.from_agraph(dot))  # Pass the Digraph object to from_agraph
    
    if plot:
        pos = nx.spring_layout(graph, seed=42)  # Consistent layout
        nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8,  # Adjusted font size
                arrowstyle='-|>', arrowsize=15)  # Improved visualization
        plt.title("Causal Graph")
        plt.show()

    return graph




def create_and_visualize_graph(graph_spec, plot=True):
    """
    Creates a directed acyclic graph (DAG) from a graphviz specification and visualizes it.

    Args:
        graph_spec: A string containing the graphviz graph specification.
        plot: If True, plots the graph using matplotlib (requires graphviz installation).

    Returns:
        A networkx.DiGraph object representing the DAG.
    """
    # Method 1: Using Source and temporary file
    with tempfile.NamedTemporaryFile(suffix='.dot') as tmp:
        tmp.write(graph_spec.encode('utf-8'))
        tmp.flush()
        graph = nx.DiGraph(nx.nx_agraph.read_dot(tmp.name))
    
    # Alternative Method 2 (uncomment if Method 1 doesn't work)
    # import pydot
    # graphs = pydot.graph_from_dot_data(graph_spec)
    # graph = nx.nx_pydot.from_pydot(graphs[0])
    
    if plot:
        pos = nx.spring_layout(graph, seed=42)  # Consistent layout
        nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8,
                arrowstyle='-|>', arrowsize=15)  # Improved visualization
        plt.title("Causal Graph")
        plt.show()

    return graph

# %%

create_and_visualize_graph(
    graph_spec= create_causal_graph_spec()
)