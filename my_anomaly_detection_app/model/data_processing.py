import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder

def load_data(file_obj):
    """
    Loads data from a file-like object (uploaded via Streamlit).
    Tries CSV first; if that fails, then Excel.
    """
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        df = pd.read_excel(file_obj)
    return df

def plot_heatmap(df_numeric, ax=None):
    """
    Plots a correlation heatmap for a DataFrame of numeric columns.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
    return ax

def clean_data(df, cols_to_drop, cat_cols_to_fill):
    """
    Cleans the DataFrame by:
      - Dropping specified columns.
      - Removing duplicates.
      - Filling missing values in key categorical columns with "Missing".
      - Dropping rows missing the essential datetime field.
    """
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    df_clean = df_clean.drop_duplicates()
    for col in cat_cols_to_fill:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Missing")
    if 'INSTRUMENTINGESTIONDATETIME' in df_clean.columns:
        df_clean = df_clean[df_clean['INSTRUMENTINGESTIONDATETIME'].notna()]
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def extract_datetime_features(df, datetime_cols):
    """
    Converts specified datetime columns to datetime objects and extracts
    year and month into new columns. Then drops the original datetime columns.
    """
    df_new = df.copy()
    for col in datetime_cols:
        if col in df_new.columns:
            df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
            df_new[f"{col}_year"] = df_new[col].dt.year
            df_new[f"{col}_month"] = df_new[col].dt.month
        else:
            print(f"Warning: Datetime column '{col}' not found.")
    df_new = df_new.drop(columns=datetime_cols, errors='ignore')
    return df_new

def encode_categoricals(df, categorical_cols):
    """
    Label-encodes specified categorical columns.
    """
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        else:
            print(f"Warning: Column '{col}' not found.")
    return df_encoded

def plot_histogram(df, column):
    """
    Plots and returns a histogram (with KDE) for a numeric column.
    """
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    return fig

def prepare_features(df, numeric_cols, categorical_cols):
    """
    Combines numeric and categorical features, applies Robust Scaling, and returns:
      - X: DataFrame of selected features.
      - X_scaled: Scaled numpy array.
      - features: List of feature names.
    """
    features = []
    for col in numeric_cols:
        if col in df.columns:
            features.append(col)
    for col in categorical_cols:
        if col in df.columns:
            features.append(col)
    X = df[features].copy()
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, features
