#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from causal_analysis_module.analysis import data_preprocessing
import pandas_flavor as pf

# %%
df = data_preprocessing()
# %%
treatment_lst = ['food_rating', 'service_rating', 'age_group', 'activity', 'personality', 'User_cuisine']
outcome_var = 'rating'

@pf.register_dataframe_method
def prepare_data_for_causalforest(df, treatment_vars, outcome_var, verbose=True):
    """
    Prepares data for CausalForestDML analysis by properly handling
    categorical and numerical features.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input data
    treatment_vars : list
        List of treatment variable names
    outcome_var : str
        Outcome variable name
    verbose : bool, default=True
        Whether to print information during processing
        
    Returns:
    --------
    dict
        Dictionary containing prepared data for CausalForestDML
    """
    if verbose:
        print("Starting data preparation for CausalForestDML...")
        print(f"Shape of input data: {df.shape}")
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # 1. Identify variable types
    # Identify categorical and numerical columns
    cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if verbose:
        print(f"Categorical columns: {len(cat_cols)}")
        print(f"Numerical columns: {len(num_cols)}")
    
    # Remove outcome and treatment from feature lists
    feature_cat_cols = [col for col in cat_cols if col not in treatment_vars + [outcome_var]]
    feature_num_cols = [col for col in num_cols if col not in treatment_vars + [outcome_var]]
    
    # 2. Handle missing values
    # Check for missing values
    missing_cols = df_copy.columns[df_copy.isnull().any()].tolist()
    
    if missing_cols:
        if verbose:
            print(f"Columns with missing values: {missing_cols}")
            print("Handling missing values...")
        
        # For numerical columns: fill with median
        for col in [c for c in missing_cols if c in num_cols]:
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        
        # For categorical columns: fill with mode
        for col in [c for c in missing_cols if c in cat_cols]:
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    else:
        if verbose:
            print("No missing values found.")
    
    # 3. Extract outcome and treatments
    Y = df_copy[outcome_var].values
    
    # Handle multiple or single treatment
    if len(treatment_vars) == 1:
        T = df_copy[treatment_vars[0]].values
    else:
        T = df_copy[treatment_vars].values
    
    # 4. Create preprocessing pipeline for features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), feature_cat_cols)
        ],
        remainder='drop'  # Drop columns not specified
    )
    
    # 5. Fit and transform the data
    try:
        X = preprocessor.fit_transform(df_copy)
        
        # Get feature names after transformation
        num_feature_names = feature_num_cols
        
        # Get one-hot encoded feature names
        try:
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(feature_cat_cols)
            all_feature_names = np.concatenate([num_feature_names, cat_feature_names])
        except:
            all_feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])])
        
        if verbose:
            print(f"Data transformed successfully.")
            print(f"Transformed feature matrix shape: {X.shape}")
            print(f"Number of features after transformation: {X.shape[1]}")
    
    except Exception as e:
        print(f"Error in data transformation: {str(e)}")
        # Fallback: just use numerical features
        X = df_copy[feature_num_cols].values
        all_feature_names = np.array(feature_num_cols)
        print("Falling back to numerical features only.")
    
    # 6. Create train-test split
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=42
    )
    
    if verbose:
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
    
    # 7. Return prepared data
    prepared_data = {
        'X': X,
        'T': T,
        'Y': Y,
        'X_train': X_train,
        'X_test': X_test,
        'T_train': T_train,
        'T_test': T_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'feature_names': all_feature_names,
        'preprocessor': preprocessor
    }
    
    return prepared_data




@pf.register_dataframe_method
def handle_categorical_treatments(df, treatment_vars):
    """
    Properly prepares categorical treatment variables for causal forest.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input data
    treatment_vars : list
        List of treatment variable names
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with properly processed treatments
    """
    df_copy = df.copy()
    
    for treatment in treatment_vars:
        # Check if treatment is categorical
        if df_copy[treatment].dtype == 'object' or df_copy[treatment].dtype.name == 'category':
            print(f"Converting categorical treatment '{treatment}' to numerical...")
            
            # For binary categorical variables, convert to 0/1
            if len(df_copy[treatment].unique()) == 2:
                # Get the categories
                categories = df_copy[treatment].unique()
                print(f"  Binary treatment with values: {categories}")
                
                # Map to 0/1
                df_copy[treatment] = df_copy[treatment].map({categories[0]: 0, categories[1]: 1})
            
            # For multi-category treatments
            else:
                print(f"  Multi-category treatment with {len(df_copy[treatment].unique())} values")
                # In this case, we might want to create multiple binary treatments
                # or use a different approach depending on the research question
                
                # For now, we'll convert to numerical codes
                df_copy[treatment] = df_copy[treatment].astype('category').cat.codes
                print(f"  Converted to categorical codes (0 to {len(df_copy[treatment].unique())-1})")
                
                # Note: This approach might not be ideal for causal inference
                # with multi-category treatments. Consider creating separate
                # binary treatments for each category if needed.
    
    return df_copy




treatment_lst = ['food_rating', 'service_rating', 'age_group', 'activity', 'personality', 'User_cuisine']

@pf.register_dataframe_method
def prepare_for_causal_analysis(df, treatment_vars, outcome_var):
    """
    Master function to prepare data for causal forest analysis.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input data
    treatment_vars : list
        List of treatment variable names
    outcome_var : str
        Outcome variable name
        
    Returns:
    --------
    dict
        Dictionary containing prepared data for CausalForestDML
    """
    print(f"Preparing data for causal analysis with CausalForestDML...")
    print(f"Treatments: {treatment_vars}")
    print(f"Outcome: {outcome_var}")
    
    # 1. Initial data examination
    print("\nInitial data examination:")
    print(f"Data shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    
    # Count unique values for each column to identify categorical variables
    n_unique = df.nunique()
    print("\nNumber of unique values in each column:")
    for col in df.columns:
        print(f"  {col}: {n_unique[col]}")
    
    # 2. Handle categorical treatments
    df_processed = handle_categorical_treatments(df, treatment_vars)
    
    # 3. Identify potentially problematic columns
    # Columns with too many unique values might cause issues with one-hot encoding
    high_cardinality_cols = [col for col in df_processed.columns 
                             if df_processed[col].dtype == 'object' and n_unique[col] > 20]
    
    if high_cardinality_cols:
        print(f"\nWarning: The following columns have high cardinality:")
        for col in high_cardinality_cols:
            print(f"  {col}: {n_unique[col]} unique values")
        print("Consider handling these columns specifically or dropping them.")
    
    # 4. Drop problematic columns if necessary
    # For example, columns that are unique identifiers or text fields
    cols_to_drop = ['userID', 'longitude', 'p.longitude', 'p.latitude', 'birth_year',
       'placeID', 'age']  # Add columns to drop here if needed
    
    if cols_to_drop:
        print(f"\nDropping problematic columns: {cols_to_drop}")
        df_processed = df_processed.drop(columns=cols_to_drop)
    
    # 5. Prepare data for CausalForestDML
    prepared_data = prepare_data_for_causalforest(
        df=df_processed,
        treatment_vars=treatment_vars,
        outcome_var=outcome_var,
        verbose=True
    )
    
    print("\nData preparation complete!")
    return prepared_data




df_prep = prepare_for_causal_analysis(df = data_preprocessing(),
                            treatment_vars = treatment_lst , outcome_var = 'rating'
                            
)

# %%
