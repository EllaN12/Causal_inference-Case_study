#%%
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pandas as pd
from pathlib import Path
import pandas_flavor as pf
from geopy.distance import geodesic
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ydata_profiling import ProfileReport
from sklearn.cluster import KMeans



# Get current directory and construct file path
"""dir_path = os.getcwd()
file_path = os.path.join(dir_path, "00_Raw_Data", "*.csv")
files_path = glob.glob(file_path)"""

#%%
@pf.register_dataframe_method
def read_csv_files():
    """
    Read all CSV files from the raw data directory and load them into DataFrames.
    
    This function automatically discovers and loads all CSV files from the '00_Raw_Data'
    directory relative to the current working directory. Each file is loaded into a
    pandas DataFrame and stored in a dictionary using the filename (without extension)
    as the key.
    
    Expected CSV files in '00_Raw_Data' directory:
        - chefmozparking.csv: Restaurant parking information
        - chefmozaccepts.csv: Payment methods accepted by restaurants
        - userpayment.csv: User payment preferences
        - geoplaces2.csv: Restaurant geographic and profile data
        - rating_final.csv: User ratings for restaurants
        - usercuisine.csv: User cuisine preferences
        - chefmozcuisine.csv: Restaurant cuisine specialties
        - chefmozhours4.csv: Restaurant operating hours
        - userprofile.csv: User demographic and preference data
    
    Returns:
        dict: Dictionary of pandas DataFrames where keys are filenames (without .csv
              extension) and values are the corresponding DataFrames. For example:
              {'chefmozparking': DataFrame, 'geoplaces2': DataFrame, ...}
    
    Raises:
        FileNotFoundError: If the '00_Raw_Data' directory does not exist
        Exception: If any CSV file fails to load (error is printed but doesn't stop execution)
    
    Example:
        >>> dataframes = read_csv_files()
        Successfully loaded: chefmozparking with shape (702, 3)
        Successfully loaded: geoplaces2 with shape (130, 21)
        ...
        >>> restaurant_df = dataframes['geoplaces2']
        >>> print(restaurant_df.head())
    """
    # Get current directory and construct path to raw data folder
    dir_path = os.getcwd()
    data_dir = os.path.join(dir_path, "00_Raw_Data")
    
    file_path = os.path.join(data_dir, "*.csv")
    files_path = glob.glob(file_path)
    
    # Check if any CSV files were found
    if not files_path:
        print(f"Warning: No CSV files found in {data_dir}")
        return {}

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
    print(f"\nAvailable dataframes: {len(dataframes)}")
    for key in dataframes.keys():
        print(f"- {key}")
    
    return dataframes






#%%
@pf.register_dataframe_method
def data_preprocessing():
    """
    Perform comprehensive data preprocessing for restaurant recommendation system.
    
    This function merges multiple data sources (user profiles, restaurant data, ratings),
    cleans and transforms categorical and numerical features, handles missing values,
    and engineers new features including distance calculations, location clustering,
    age groups, and cuisine matching scores.
    
    Returns:
        pd.DataFrame: Preprocessed dataset with the following key features:
            - User demographic data (age, age_group, smoker, etc.)
            - Restaurant information (name, location, cuisine, ambience, etc.)
            - Ratings and payment preferences
            - Engineered features:
                * patrons_restaurant_distance: Distance in kilometers
                * location_cluster: Geographic cluster (0-4)
                * age_group: Categorized age ranges
                * cuisine_match_score: User-restaurant cuisine similarity (0-1)
                * Business_hours: Categorized operating hours
    
    Processing steps:
        1. Load and merge patron and restaurant datasets
        2. Separate categorical and numerical columns
        3. Clean categorical data (replace missing markers, standardize values)
        4. Parse and categorize business hours
        5. Impute missing values using most frequent strategy
        6. Engineer distance, clustering, age, and cuisine matching features
    """
    
    dataframes = read_csv_files()

    chefmozparking_df = dataframes['chefmozparking']
    chefmozaccepts_df = dataframes['chefmozaccepts']
    userpayment_df = dataframes['userpayment']
    restaurant_geoplaces2_df = dataframes['geoplaces2']
    user_rating_final = dataframes['rating_final']
    usercuisine_df = dataframes['usercuisine']
    chefmozcuisine_df = dataframes['chefmozcuisine']
    chefmozhours4_df = dataframes['chefmozhours4']
    userprofile_df = dataframes['userprofile']

    # Merge patron-related data
    patrons_df = pd.merge(userprofile_df, userpayment_df, on='userID', how='left')
    patrons_df = patrons_df \
        .merge(usercuisine_df, on='userID', how='left') \
        .merge(user_rating_final, on='userID', how='left')
    
    patrons_df.rename(columns={'latitude': 'p.latitude', 'longitude': 'p.longitude'}, inplace=True)
    
    # Merge restaurant-related data
    restaurant_df = pd.merge(restaurant_geoplaces2_df, chefmozaccepts_df, on='placeID', how='left') 
    restaurant_df = restaurant_df\
        .merge(chefmozparking_df, on='placeID', how='left') \
        .merge(chefmozcuisine_df, on='placeID', how='left') \
        .merge(chefmozhours4_df, on='placeID', how='left') 
    
    # Combine patron and restaurant data
    data = patrons_df.merge(restaurant_df, on='placeID', how='left')
    df = data

    # Separate categorical and numerical columns for processing
    category_columns = df.select_dtypes(include=['object']).columns.tolist()
    cat_data_df = df[category_columns]
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_data_df = df[numerical_columns]

    # Preprocessing categorical data
    cat_data_df.loc[:, 'smoker'] = cat_data_df['smoker'].replace('?', np.nan)
    cat_data_df.loc[:, 'days'] = cat_data_df['days'].replace('Mon;Tue;Wed;Thu;Fri;', 'weekdays')
    cat_data_df['dress_preference'] = cat_data_df['dress_preference'].replace('?', np.nan)
    cat_data_df['ambience'] = cat_data_df['ambience'].replace('?', np.nan)
    cat_data_df.rename(columns={
        'Rcuisine_x': 'User_cuisine', 
        'name': 'restaurant_name', 
        'Rpayment': 'restaurant_payment', 
        'Rcuisine_y': 'restaurant_specialty',
        'Rambience':'ambiance'
    }, inplace=True)

    # Remove unwanted columns (FIXED: reassign result)
    cat_data_df = cat_data_df.drop(columns=['fax'])

    # Handle time data
    cat_data_df['hours'] = cat_data_df['hours'].astype(str)

    # Remove trailing semicolon and split into start and end times
    cat_data_df['start_time'] = cat_data_df['hours'].str[:-1].str.split('-').str[0]
    cat_data_df['end_time'] = cat_data_df['hours'].str[:-1].str.split('-').str[1]

    # Convert to datetime
    cat_data_df['start_time'].replace('na', np.nan, inplace=True)
    cat_data_df['end_time'].replace('na', np.nan, inplace=True)

    cat_data_df['start_time'] = pd.to_datetime(cat_data_df['start_time'], format='%H:%M')
    cat_data_df['end_time'] = pd.to_datetime(cat_data_df['end_time'], format='%H:%M')

    def categorize_time(row):
        """
        Categorize restaurant operating hours into time periods.
        
        Args:
            row (pd.Series): DataFrame row containing 'start_time' and 'end_time' columns
        
        Returns:
            str: Time category - one of:
                - 'Invalid': Missing time data
                - 'Morning': 6:00 AM - 12:00 PM
                - 'Afternoon': 12:00 PM - 6:00 PM
                - 'Evening': 6:00 PM onwards or spans into early morning
                - '24H': 24-hour operation (end time before start time)
                - 'Full Day': Spans from morning through evening
                - 'Night': Other nighttime operations
        """
        if pd.isna(row['start_time']) or pd.isna(row['end_time']):
            return 'Invalid'

        start_time = row['start_time']
        end_time = row['end_time']

        if start_time >= pd.to_datetime('06:00').time() and end_time <= pd.to_datetime('12:00').time():
            return 'Morning'
        elif start_time >= pd.to_datetime('12:00').time() and start_time < pd.to_datetime('18:00').time():
            return 'Afternoon'
        elif start_time >= pd.to_datetime('18:00').time() or (start_time < pd.to_datetime('06:00').time() and end_time >= pd.to_datetime('06:00').time()):
            return 'Evening'
        elif end_time < start_time:  # 24-hour operation
            return '24H'
        elif start_time >= pd.to_datetime('06:00').time() and end_time >= pd.to_datetime('18:00').time():
            return 'Full Day'
        else:
            return 'Night'
        
    def process_business_hours(df):
        """
        Process and categorize business hours for restaurants.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'start_time' and 'end_time' columns
        
        Returns:
            pd.DataFrame: Input DataFrame with added 'Business_hours' column containing
                         categorized time periods, and start/end times converted to time objects
        """
        df.replace('na', np.nan, inplace=True)
        df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M').dt.time
        df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M').dt.time
        df['Business_hours'] = df.apply(categorize_time, axis=1)
        return df
    
    cat_data_df = process_business_hours(cat_data_df)
       
    # Identify columns with missing values
    columns_with_missing_values_cat = [column for column in cat_data_df.columns if cat_data_df[column].isnull().any()]
    columns_with_missing_values_num = [column for column in numerical_data_df.columns if numerical_data_df[column].isnull().any()]
    
    # Impute missing categorical values with most frequent value
    imputer = SimpleImputer(strategy="most_frequent")
    cat_data_df[columns_with_missing_values_cat] = imputer.fit_transform(cat_data_df[columns_with_missing_values_cat]) 
    
    # Combine numerical and categorical data
    final_data_df = pd.concat([numerical_data_df, cat_data_df], axis=1)

    # FEATURE ENGINEERING
    
    def calculate_distance(row):
        """
        Calculate geodesic distance between user and restaurant locations.
        
        Args:
            row (pd.Series): DataFrame row containing:
                - 'p.latitude', 'p.longitude': User coordinates
                - 'latitude', 'longitude': Restaurant coordinates
        
        Returns:
            float: Distance in kilometers between user and restaurant
        """
        user_loc = (row['p.latitude'], row['p.longitude'])
        rest_loc = (row['latitude'], row['longitude'])
        return geodesic(user_loc, rest_loc).kilometers

    # Calculate patron-restaurant distance
    final_data_df['patrons_restaurant_distance'] = final_data_df.apply(calculate_distance, axis=1)  
            
    # Create restaurant location clusters based on geographic coordinates
    coords = final_data_df[['latitude', 'longitude']].values
    final_data_df['location_cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(coords)
    
    # Create age groups from birth year
    current_year = 2025
    final_data_df['age'] = current_year - final_data_df['birth_year']
    final_data_df['age_group'] = pd.cut(final_data_df['age'], 
                                bins=[0, 25, 35, 50, 100],
                                labels=['18-25', '26-35', '36-50', '50+'])

    def calculate_cuisine_match(row):
        """
        Calculate cuisine matching score between user preferences and restaurant offerings.
        
        Args:
            row (pd.Series): DataFrame row containing:
                - 'User_cuisine': Comma-separated user cuisine preferences
                - 'restaurant_specialty': Comma-separated restaurant cuisines
        
        Returns:
            float: Matching score (0-1) representing the proportion of user's preferred
                   cuisines that match restaurant offerings. Returns 0 if no data available.
        """
        user_cuisines = row['User_cuisine'].split(',') if isinstance(row['User_cuisine'], str) else []
        rest_cuisines = row['restaurant_specialty'].split(',') if isinstance(row['restaurant_specialty'], str) else []
        
        # Calculate matching score
        if user_cuisines and rest_cuisines:
            matches = len(set(user_cuisines) & set(rest_cuisines))
            total = len(set(user_cuisines))
            return matches / total if total > 0 else 0
        return 0
        
    final_data_df['cuisine_match_score'] = final_data_df.apply(calculate_cuisine_match, axis=1)
    
    return final_data_df




# %%
