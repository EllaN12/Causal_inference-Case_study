import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from causal_analysis_module.analysis import data_preprocessing
from causal_analysis_module.causal_graph_module_complete import run_complete_causal_analysis, print_graph_summary , correlation_edges, get_causal_paths, get_confounders
from causal_analysis_module.mediation_analysis_module import run_mediation_analysis
from causal_analysis_module.HTE_analysis import run_HTE_analysis
import os
import pathlib

#%%
df = data_preprocessing()
print(df.columns)
#causal_graph = create_causal_graph_spec()

treatments = ['color', 'height', 'hijos', 'interest', 'personality', 'food_rating', 'service_rating']
outcome = 'rating'

#%%
graph, treatments = run_complete_causal_analysis()
print_graph_summary(graph, outcome=outcome)
Hijos_paths = get_causal_paths(graph, 'hijos', 'rating')
food_rating_paths = get_causal_paths(graph, 'food_rating', 'rating')
service_paths = get_causal_paths(graph, 'service_rating', 'rating')
hijos_confounders = get_confounders(graph, 'hijos', 'rating')
print(Hijos_paths)
print(food_rating_paths)
print(service_paths)
print(hijos_confounders)
# %%
print (df['restaurant_specialty'].unique())

#%%

