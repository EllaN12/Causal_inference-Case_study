
# Libraries and Packages
#%%

# Data Manipulation:
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
#import funnelplot
import plotly.express as px
from ydata_profiling import ProfileReport, profile_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ydata_profiling import ProfileReport
import seaborn as sns
from pathlib import Path
from causal_analysis_module.analysis import data_preprocessing




# Modeling:
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
from sklearn.metrics import roc_auc_score


##Questions :----
# - What Causes a restaurant to be highly rated?
# What can restaurants do to improve their ratings?

# Data

import os
import glob
import pandas as pd
from pathlib import Path



df = data_preprocessing()

#%%

#patrons_df = pd.merge(patrons_df, user_cuisine_df, on='userID', how='left')
 
def read_csv_files(files_path):
    """AI is creating summary for read_csv_files

    Args:
        00_Raw_Data ([.csv]): The path to the raw data

    Returns:
        [pandas.dataframe]: panadas DataFrames used to perform further analyisi.
    """
    dataframes = {}
    for file in files_path:
        filename = Path(file).stem
        try:
            df = pd.read_csv(file)
            dataframes[filename] = df
            print(f"Successfully loaded: {filename} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    # Print available dataframes
    print("\nAvailable dataframes:")
    for key in dataframes.keys():
        print(f"- {key}")
    
    return dataframes





dataframes = read_csv_files(files_path)








chefmozparking_df = dataframes['chefmozparking']  
chefmozaccepts_df = dataframes['chefmozaccepts']
userpayment_df = dataframes['userpayment']
restaurant_geoplaces2_df = dataframes['geoplaces2']
user_rating_final = dataframes['rating_final']
usercuisine_df = dataframes['usercuisine']
chefmozcuisine_df = dataframes['chefmozcuisine']
chefmozhours4_df = dataframes['chefmozhours4']
userprofile_df = dataframes['userprofile']

# Accessing DataFrame and filename for each tuple in the list


# Merging dataframes in users and restaurants
patrons_df = pd.merge(userprofile_df,userpayment_df , on='userID', how='left')
patrons_df = patrons_df \
    .merge(usercuisine_df, on='userID', how='left') \
    .merge(user_rating_final, on='userID', how='left')

patrons_df

restaurant_df = pd.merge(restaurant_geoplaces2_df, chefmozaccepts_df, on='placeID', how='left') 

restaurant_df = restaurant_df\
    .merge(chefmozparking_df, on='placeID', how='left') \
    .merge(chefmozcuisine_df, on='placeID', how='left') \
    .merge(chefmozhours4_df, on='placeID', how='left') 


restaurant_df
#renaming geo_locations columns on patrons dataframe
patrons_df.rename(columns={'latitude':'p.latitude', 'longitude':'p.longitude' }, inplace=True)
 

#Data Exploration and Cleaning

# %%


rating_count = patrons_df['rating'].value_counts()
rating_count

# How many people rated restaurants?
users_count = patrons_df['userID'].unique().shape
users_count
# saving the raw data
#patrons_df.to_pickle('/Users/ellandalla/Desktop/Causal_inference-Case_study/venv/00_Raw_Data/patrons_raw.pkl')
#restaurant_df.to_pickle('/Users/ellandalla/Desktop/Causal_inference-Case_study/venv/00_Raw_Data/restaurant_raw.pkl')



patrons_df.describe()
# Exploratory data analysis


# %%

#Merge both Patrons and Restaurant dataframes
data = patrons_df\
    .merge(restaurant_df, on='placeID', how='left')
    
    
data['area'].unique()

# Pandas profling on the full dataset
profile = data.profile_report()
output_path = os.path.join(dir_path, "Reports", "data_report.html")
profile.to_file(output_path)

#Segregate the columns into numerical and categorical

columns = data.columns.tolist()

print(columns)

numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
numerical_columns

category_columns = data.select_dtypes(include = ['object']).columns.tolist()
category_columns


numerical_data_df = data[numerical_columns]
numerical_data_df


def profile_data(numerical_data_df):
    """Panda Profiling Function

    Args:
        data (DataFrame): A data frame to profile

    Returns:
        DataFrame: A data frame with profiled data
    """
    return pd.concat(
        [
            pd.Series(numerical_data_df.dtypes, name = "Dtype"),
            # Counts
            pd.Series(numerical_data_df.count(), name = "Count"),
            pd.Series(numerical_data_df.isnull().sum(), name = "NA Count"),
            pd.Series(numerical_data_df.nunique(), name = "Count Unique"),
            # Stats
            pd.Series(numerical_data_df.min(), name = "Min"),
            pd.Series(numerical_data_df.max(), name = "Max"),
            pd.Series(numerical_data_df.mean(), name = "Mean"),
            pd.Series(numerical_data_df.median(), name = "Median"),
            pd.Series(numerical_data_df.mode().iloc[0], name = "Mode"),
        ],
        axis=1
    )
    
profile_data(numerical_data_df)


#filter geo location columns
geo_location_cols = ['p.latitude', 'p.longitude', 'latitude', 'longitude']

numerical_cols = [col for col in numerical_columns if col not in geo_location_cols]

numerical_cols.remove('placeID')
# histogram for numerical columns
%matplotlib inline
num_fig = plt.figure(figsize=(20, 10))
for i in range(len(numerical_cols)):
    num_fig.add_subplot(2, 5, i+1)
    plt.hist(numerical_data_df[numerical_cols[i]])
    plt.title(numerical_cols[i])
    plt.show()
    
    
#Category data exploratory and cleaning
#%%
categoty_data_df = data[category_columns]
categoty_data_df.head()

#Preprocessing Categorical Data
#%%

categoty_data_df['smoker'].unique()
categoty_data_df.loc[:, 'smoker'] = categoty_data_df['smoker'].replace('?', np.nan)
categoty_data_df.loc[:, 'days']= categoty_data_df['days'].replace('Mon;Tue;Wed;Thu;Fri;', 'weekdays')

categoty_data_df['drink_level'].unique()

categoty_data_df['dress_preference'].unique()
categoty_data_df['dress_preference']= categoty_data_df['dress_preference'].replace('?', np.nan)

categoty_data_df['ambience'].unique()
categoty_data_df['ambience']= categoty_data_df['ambience'].replace('?', np.nan)

categoty_data_df.rename(columns={'Rcuisine_x':'User_cuisine', 'name':'restaurant_name', 'Rpayment': 'restaurant_payment', 'Rcuisine_y': 'restaurant_specialty'}, inplace=True)
categoty_data_df.replace('?', np.nan, inplace=True)

categoty_data_df['url'].unique()

#Removing unwanted columns
categoty_data_df.drop(columns = ['fax'])

categoty_data_df['hours'].unique()


categoty_data_df['url'].unique()


#handling Time

categoty_data_df['hours'] = categoty_data_df['hours'].astype(str)

# Remove the trailing semicolon and split into start and end times
categoty_data_df['start_time'] = categoty_data_df['hours'].str[:-1].str.split('-').str[0]
categoty_data_df['end_time'] = categoty_data_df['hours'].str[:-1].str.split('-').str[1]

#convert to datetime
categoty_data_df['start_time'].replace('na', np.nan, inplace=True)
categoty_data_df['end_time'].replace('na', np.nan, inplace=True)


categoty_data_df['start_time'] = pd.to_datetime (categoty_data_df['start_time'], format = '%H:%M')
categoty_data_df['end_time'] = pd.to_datetime (categoty_data_df['end_time'], format = '%H:%M')


categoty_data_df[['start_time', 'end_time']]
 
df = categoty_data_df 
df
# define categories for time of day
# Define time categories
Business_hours = {
    'Morning': ((df['start_time'] >=('06:00')) & 
                (df['start_time'] < ('12:00'))),
    'Afternoon': ((df['start_time'] >= ('12:00')) & 
                  (df['start_time'] < ('18:00'))),
    'Evening': ((df['start_time'] >= ('18:00')) & 
                (df['start_time'] < ('00:00'))),
    'Night': ((df['start_time'] >= ('00:00')) | 
              (df['start_time'] < ('06:00'))),
    "24H" : ((df['start_time'] >= ('00:00')) &
             (df['start_time'] < ('00:00')))
             
}
#Convert start_time and end_time to datetime.time
df.replace('na', np.nan, inplace=True)
df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M').dt.time
df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M').dt.time


def categorize_time(row):
    
    """_function to classify restaurant hours into time categories_

    Returns:
        _function_: _return classsification into a new column " category"_
    """
    
    if pd.isna(row['start_time']) or pd.isna(row['end_time']):
        return 'Invalid'
    start_time = row['start_time']
    end_time = row['end_time']
    
    if row['start_time'] >= pd.to_datetime('06:00').time() and row['end_time'] <= pd.to_datetime('12:00').time():
        return 'Morning'
    elif row['start_time'] >= pd.to_datetime('12:00').time() and row['start_time'] < pd.to_datetime('18:00').time():
        return 'Afternoon'
    elif row['start_time'] >= pd.to_datetime('18:00').time() and row['start_time'] < pd.to_datetime('00:00').time():
        return 'Evening'
    elif (row['start_time'] >= pd.to_datetime('00:00').time() or row['start_time'] < pd.to_datetime('06:00').time()) and row['end_time'] == pd.to_datetime('23:30').time():
        return '24H'
    elif row['start_time'] >= pd.to_datetime('06:00').time() and row['start_time'] < pd.to_datetime('12:00').time() and row['end_time'] <= pd.to_datetime('23:30').time():
        return 'Full Day'
    else:
        return 'Night'
    
df['Business_hours'] = df.apply(categorize_time, axis=1)
df.head()


# replace to NaN, hours that are invalid
df['Business_hours'].replace('Invalid', np.nan, inplace=True)


df['Business_hours'].unique()
# %%
# keep necessary categorical columns
columns_to_keep = [#'userID', 
                    'smoker', 'drink_level', 'dress_preference', 'ambience',
       'transport', 'marital_status', 'hijos', 'interest', 'personality',
       'religion', 'activity', 'color', 'budget', 'Upayment', 'User_cuisine',
       'the_geom_meter', 'restaurant_name', 'address', 'city', 'state',
       #'country', 'fax', 
       'zip', 
       'alcohol', 'smoking_area', 'dress_code',
       'accessibility', 'price', 'url', 'Rambience', 'franchise', 'area',
       'other_services', 'restaurant_payment', 'parking_lot',
       'restaurant_specialty', #'hours', 'days', 'start_time', 'end_time',
       'Business_hours']
      


value_counts = df['Business_hours'].value_counts()
value_counts

category_df = df[columns_to_keep]
category_columns = category_df.columns.tolist()

#%%
#Visualizing Categorical Data
%matplotlib inline
cat_fig = plt.figure(figsize=(40, 40))
for i in range(len(columns_to_keep)):
    ax = cat_fig.add_subplot(7, 5, i+1)
    value_counts = category_df[columns_to_keep[i]].value_counts()
    ax.bar(value_counts.index, value_counts.values)
    ax.set_title(columns_to_keep[i])
plt.tight_layout()
plt.show()
    
# Addressing Missing values
#%%
# Check for missing values in the numerical data
numerical_data_df
columns_with_missing_values_num = [column for column in numerical_data_df.columns if numerical_data_df[column].isnull().any()]
columns_with_missing_values_num


category_df
columns_with_missing_values_cat = [column for column in category_df.columns if category_df[column].isnull().any()]
columns_with_missing_values_cat
    
        
#Initialize SimpleImputer with the desired strategy
imputer = SimpleImputer(strategy="most_frequent")

# Apply imputer to columns with missing values
category_df[columns_with_missing_values_cat] = imputer.fit_transform(category_df[columns_with_missing_values_cat])


## Merge both numerical and categorical data as final df

final_data_df = pd.concat([numerical_data_df, category_df], axis=1)


# Missing values Crosscheck
columns_with_missing_values = [column for column in final_data_df if final_data_df[column].isnull().any()]
columns_with_missing_values

path_dir = os.getcwd()
final_data_df.to_pickle(os.path.join(path_dir,"01_Data_analysis", "final_data.pkl")) 

print(final_data_df)
print (final_data_df.columns.tolist())

"""
'p.latitude', 'p.longitude', 'birth_year', 'weight', 'height', 
'placeID', 'rating', 'food_rating', 'service_rating', 'latitude',
'longitude', 'smoker', 'drink_level', 'dress_preference', 'ambience', 
'transport', 'marital_status', 'hijos', 'interest', 'personality', 
'religion', 'activity', 'color', 'budget', 'Upayment', 'User_cuisine', 
'the_geom_meter', 'restaurant_name', 'alcohol', 'smoking_area', 'dress_code', 
'accessibility', 'price', 'url', 'Rambience', 'franchise', 'area', 'other_services', 
'restaurant_payment', 'parking_lot', 'restaurant_specialty', 'Business_hours'

"""
#%%
# Building KPIs 

# Find mean ratings across data sets

mean_meadian_rating  = final_data_df.agg({'rating': ['mean', 'median'], 'food_rating': ['mean', 'median'], 'service_rating': ['mean', 'median']})


# Group the data by various columns

# Group the data by restaurant attriutes
Restaurant_grouped_final_data = final_data_df.groupby(['placeID','alcohol', 'smoking_area', 'dress_code','accessibility', 'price', 'Rambience', 'area', 
                                            'parking_lot', 'restaurant_specialty', 'Business_hours'])

# Group the data by user attributes
user_grouped_final_data = final_data_df.groupby(['birth_year','smoker', 'drink_level', 'dress_preference', 'ambience', 'transport', 'marital_status', 'hijos', 'interest', 'personality', 
                                                 'religion', 'activity', 'color', 'budget', 'Upayment', 'User_cuisine'])

final_data_df['birth_year'].unique()

# Aggregate the data = Resturant 
restaurant_data_summary = Restaurant_grouped_final_data.agg({'rating': ['mean', 'median'], 'food_rating': ['mean', 'median'], 'service_rating': ['mean', 'median']})

restaurant_rating = Restaurant_grouped_final_data.agg({'rating': ['mean', 'median']})


# Aggregate the data = User Data
user_data_summary = user_grouped_final_data.agg({'rating': ['mean', 'median'], 'food_rating': ['mean', 'median'], 'service_rating': ['mean', 'median']})
















user_data_binarized = user_data_summary.binarize()



user_data_correlated = user_data_summary.correlate(target='rating')



restuarant_df  = final_data_df[['placeID','alcohol', 'smoking_area', 'dress_code','accessibility', 'price', 'Rambience', 'area', 
                                            'parking_lot', 'restaurant_specialty', 'Business_hours']] 




#%%
#Encoding and Scaling Data 
# Categorical data

"""
np_array = category_df.to_numpy()

encoder = OneHotEncoder(dtype=np.uint8)
X_encoded = encoder.fit_transform(np_array)
encoded_cat_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(category_columns))
from sklearn.preprocessing import OneHotEncoder

# print categorical columns
print (category_df.columns.tolist())

# Numerical data
print(numerical_data_df.columns.tolist())


"""" numerical_data_df = ['p.latitude', 'p.longitude', 'birth_year', 'weight', 'height', 'rating', 'food_rating', 'service_rating', 'latitude', 'longitude']

"""scaled_num_df = pd.DataFrame(RobustScaler().fit_transform(numerical_data_df), columns=numerical_data_df.columns) """


# Combine the encoded categorical and scaled numerical data
""" encoded_final_df = pd.concat([scaled_num_df, encoded_cat_df], axis=1)"""
#save final analysis 

#final_data_df.to_pickle('/Users/ellandalla/Desktop/Causal_inference-Case_study/venv/01_Data_analysis/final_data.pkl')

""""

# 2.0 CORRELATION (LEVEL 1: ASSOCIATION) ----
#%%
target_column = "rating"
other_columns = list(encoded_final_df.columns)
other_columns.remove(target_column)  # Exclude target column

filtered_df = encoded_final_df[[target_column] + other_columns]




#df_correlated = filtered_df.correlate(target="rating")

df_correlated = filtered_df.corr()

# Melting the DataFrame to long format
#df_long = df_correlated.melt(id_vars= df_correlated.columns, var_name='correlated_feature', value_name='correlation')

# If you want to visualize specific columns
#other_columns = df_long['correlated_feature'].unique()

# Create the funnel plot using Seaborn

#%%
fig = px.funnel(
    df_long,
    x=target_colum,
    y=other_columns,
    color=target_column  # Use target column for color (adjust as needed)
    labels
)
fig.show()

?px.funnel

print(filtered_df_sorted)

# %%

plt.figure(figsize=(10, 8))
sns.heatmap(df_correlated, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

"""


# %%


