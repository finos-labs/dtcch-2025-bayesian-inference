
---

### C. `app.py`

This is the main frontend file using Streamlit. Create an `app.py` file with the following code:

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Import our custom modules
from model import data_processing, anomaly_detection, visualization

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

st.title("Anomaly Detection Dashboard")
st.write("Upload your Security Reference Data file (CSV or Excel) to run the anomaly detection pipeline.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Load data using our module function (it accepts a file-like object)
    df = data_processing.load_data(uploaded_file)
    
    st.subheader("Initial Data Exploration")
    st.write("**DataFrame Information:**")
    st.text(df.info(verbose=True))
    st.write("**Missing Values:**")
    st.write(df.isna().sum())
    st.write("**Descriptive Statistics:**")
    st.write(df.describe(include="all"))

    # Display a correlation heatmap if there are numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        st.subheader("Correlation Heatmap (Numeric Features)")
        fig, ax = plt.subplots(figsize=(10, 6))
        data_processing.plot_heatmap(df[num_cols], ax=ax)
        st.pyplot(fig)

    # --- Data Cleaning ---
    st.subheader("Data Cleaning")
    # Define which columns to drop and which categorical columns to fill
    cols_to_drop = [
        'Instrument Name', 'Tradeverse ID', 'Short Name', 'ISIN', 'CUSIP', 
        'SEDOL', 'SYMBOL', 'CFI', 'BLOOMBERG', 'Issuer LEI', 'Data Source', 
        'INSTRUMENTID', 'INSTRUMENTSEQUENCE', 'INSTRUMENTISUNDERLYER', 
        'INSTRUMENTISINDEX', 'INSTRUMENTLASTUPDATED'
    ]
    cat_cols_for_fill = ['Primary Asset Class', 'Secondary Asset Class',
                         'Instrument Currency', 'Issuer Country', 'Trading Venue',
                         'Venue Country', 'INSTRUMENTIDENTIFIERTYPE', 
                         'INSTRUMENTSOURCETYPE', 'INSTRUMENTSTATUS']
    
    df_clean = data_processing.clean_data(df, cols_to_drop, cat_cols_for_fill)
    st.write(f"After cleaning, data shape: {df_clean.shape}")

    # --- Datetime Feature Extraction ---
    st.subheader("Datetime Feature Extraction")
    datetime_cols = ['INSTRUMENTINGESTIONDATETIME']  # Only one essential datetime column
    df_clean = data_processing.extract_datetime_features(df_clean, datetime_cols)
    st.write("Data with extracted datetime features:")
    st.write(df_clean.head())

    # --- Feature Configuration ---
    numeric_cols = ['INSTRUMENTINGESTIONDATETIME_year', 'INSTRUMENTINGESTIONDATETIME_month']
    categorical_cols = ['Primary Asset Class', 'Secondary Asset Class', 'Instrument Currency',
                        'Issuer Country', 'Trading Venue', 'Venue Country', 
                        'INSTRUMENTIDENTIFIERTYPE', 'INSTRUMENTSOURCETYPE', 'INSTRUMENTSTATUS']

    # Encode categorical variables
    df_encoded = data_processing.encode_categoricals(df_clean, categorical_cols)
    
    st.subheader("Numeric Distributions")
    for col in numeric_cols:
        fig = data_processing.plot_histogram(df_encoded, col)
        st.pyplot(fig)

    # --- Feature Preparation ---
    X, X_scaled, feature_list = data_processing.prepare_features(df_encoded, numeric_cols, categorical_cols)
    st.write(f"Features used for modeling: {feature_list}")

    # --- Determine Optimal Number of Clusters ---
    st.subheader("Determining Optimal Number of Clusters")
    optimal_k, score_dict = anomaly_detection.find_optimal_k(X_scaled, k_min=2, k_max=10)
    st.write(f"Optimal number of clusters: {optimal_k}")
    st.write("Silhouette Scores:", score_dict)

    # --- Anomaly Detection ---
    st.subheader("Anomaly Detection")
    df_results = anomaly_detection.detect_anomalies_kmeans_advanced(X_scaled, df_encoded, optimal_k, n_mad=3)
    st.write("Top 10 Anomalies:")
    st.write(df_results.sort_values('distance_to_center', ascending=False).head(10))

    # --- Visualization ---
    st.subheader("Clusters and Anomalies Visualization")
    fig2 = visualization.visualize_clusters(df_results, feature_list, use_pca=True)
    st.pyplot(fig2)

    # --- Save Results (optional) ---
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results CSV", data=csv, file_name="anomaly_detection_results.csv", mime="text/csv")
else:
    st.info("Awaiting file upload.")
