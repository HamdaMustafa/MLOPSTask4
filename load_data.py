import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Normalize column names: strip leading/trailing spaces, replace spaces with underscores, and convert to lowercase
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    print("Columns in the dataset after normalization:", df.columns.tolist())  # Debug: Print the DataFrame columns after normalization
    
    # Ensure the 'gender' column exists in the dataset
    if 'gender' not in df.columns:
        raise ValueError("Column 'gender' not found in the dataset. Available columns: {}".format(df.columns.tolist()))
    
    # Separate features and the target variable
    X = df.drop('gender', axis=1)
    y = df['gender'].map({'male': 1, 'female': 0}).values  # Binary encoding and conversion to numpy array
    
    return X, y

def preprocess_data(X):
    # Specify numerical and categorical features
    numeric_features = ['age', 'height_(cm)', 'weight_(kg)', 'income_(usd)']
    categorical_features = ['occupation', 'education_level', 'marital_status', 'favorite_color']
    
    # Create a ColumnTransformer to apply different preprocessing to numeric and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='drop'  # Drop other columns that are not specified in numeric_features or categorical_features
    )
    
    # Fit the preprocessor and transform the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    return X_preprocessed
