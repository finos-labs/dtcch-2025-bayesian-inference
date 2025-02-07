# =============================================================================
# STEP 1: Install Necessary Libraries
# =============================================================================
!pip install pandas openpyxl scikit-learn matplotlib seaborn --quiet

# =============================================================================
# STEP 2: Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

# Ensure plots show inline (for Colab/Jupyter)
%matplotlib inline

# =============================================================================
# STEP 3: Define Helper Functions for the Pipeline
# =============================================================================

def load_data_from_upload():
    """
    Prompts the user to upload a CSV or XLSX file and returns the loaded DataFrame.
    """
    from google.colab import files
    print("Please upload your CSV or XLSX file now...")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    print(f"Uploaded file: {filename}")
    # Try CSV first, then Excel if error occurs
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print("Failed to read as CSV, trying Excel...")
        df = pd.read_excel(filename)
    return df

def initial_exploration(df):
    """
    Displays basic DataFrame info, missing values, descriptive statistics,
    and a correlation heatmap for numeric features.
    """
    print("\n--- DataFrame Information ---")
    print(df.info())

    print("\n--- Missing Values Per Column ---")
    print(df.isna().sum())

    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap of Numeric Features")
    plt.show()

def clean_data(df, cols_to_drop):
    """
    Cleans the data by dropping unnecessary columns, duplicates, and rows with missing values.
    """
    # Drop columns that are not needed
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    # Remove duplicate rows (if any)
    df_clean = df_clean.drop_duplicates()
    # Drop rows with missing values
    df_clean = df_clean.dropna().reset_index(drop=True)
    return df_clean

def encode_categoricals(df, categorical_cols):
    """
    Encodes specified categorical columns using label encoding.
    """
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        else:
            print(f"Warning: Categorical column '{col}' not found in the data.")
    return df_encoded

def visualize_numeric_distributions(df, numeric_cols):
    """
    Plots histograms with KDE for each numeric feature.
    """
    for col in numeric_cols:
        if col in df.columns:
            plt.figure()
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
        else:
            print(f"Warning: Numeric column '{col}' not found in the DataFrame.")

def prepare_features(df, numeric_cols, categorical_cols):
    """
    Selects and combines numeric and categorical features, then applies Robust Scaling.
    Returns the original feature DataFrame, the scaled feature array, and the feature names list.
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

def find_optimal_k(X_scaled, k_min=2, k_max=10):
    """
    Determines the optimal number of clusters (k) for KMeans using the silhouette score.
    Returns the k value with the highest silhouette score.
    """
    scores = {}
    for k in range(k_min, k_max+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"Silhouette Score for k={k}: {score:.4f}")
    optimal_k = max(scores, key=scores.get)
    print(f"Optimal number of clusters determined: k={optimal_k}")
    return optimal_k, scores

def detect_anomalies_kmeans_advanced(X_scaled, original_df, optimal_k, n_mad=3):
    """
    Performs KMeans clustering with the provided optimal_k.
    Then, for each cluster, computes the Euclidean distances from the cluster center.
    Within each cluster, anomalies are flagged if their distance exceeds:
        threshold = median(distance) + n_mad * MAD(distance)
    (MAD: Median Absolute Deviation)

    Parameters:
        X_scaled: Scaled feature array (numpy array)
        original_df: Original DataFrame (to append clustering results)
        optimal_k: Number of clusters for KMeans
        n_mad: The multiplier for MAD to set the anomaly threshold (default 3)

    Returns:
        A DataFrame with added cluster labels, distance, and anomaly flags.
    """
    # Run KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_

    # Compute Euclidean distances from each sample to its cluster center
    distances = np.linalg.norm(X_scaled - centers[cluster_labels], axis=1)

    # Initialize an array for anomaly flags
    anomaly_flags = np.zeros(len(distances), dtype=bool)

    # For each cluster, compute median and MAD, then flag anomalies
    for cluster in range(optimal_k):
        cluster_mask = cluster_labels == cluster
        cluster_distances = distances[cluster_mask]
        median_val = np.median(cluster_distances)
        mad = np.median(np.abs(cluster_distances - median_val))
        threshold = median_val + n_mad * mad
        # Mark anomalies in the cluster: distance > threshold
        anomaly_flags[cluster_mask] = cluster_distances > threshold
        print(f"Cluster {cluster}: median={median_val:.4f}, MAD={mad:.4f}, threshold={threshold:.4f}, "
              f"anomalies={np.sum(cluster_distances > threshold)}")

    print(f"\nTotal anomalies detected: {np.sum(anomaly_flags)} out of {len(distances)} samples")

    # Append results to the original DataFrame
    df_results = original_df.copy()
    df_results['cluster_label'] = cluster_labels
    df_results['distance_to_center'] = distances
    df_results['anomaly_flag'] = anomaly_flags
    return df_results

def visualize_clusters(df_results, features, use_pca=True):
    """
    Visualizes clusters and anomalies.
    If use_pca is True, projects the data into 2D using PCA.
    Cluster centers (if computed) and anomalies are highlighted.
    """
    if use_pca:
        # Use only the selected features for PCA projection
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(df_results[features])
        df_results['pca1'] = X_pca[:, 0]
        df_results['pca2'] = X_pca[:, 1]

        plt.figure(figsize=(10, 6))
        # Plot inliers by cluster
        sns.scatterplot(data=df_results, x='pca1', y='pca2', hue='cluster_label',
                        palette='viridis', alpha=0.6, legend='full')
        # Overlay anomalies with distinct marker and color
        anomalies = df_results[df_results['anomaly_flag']]
        sns.scatterplot(data=anomalies, x='pca1', y='pca2', color='red',
                        marker='X', s=100, label='Anomaly')
        plt.title("Clusters and Anomalies (PCA Projection)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.show()
    else:
        if len(features) >= 2:
            plt.figure(figsize=(10,6))
            sns.scatterplot(data=df_results, x=features[0], y=features[1], hue='cluster_label',
                            palette='viridis', alpha=0.6, legend='full')
            anomalies = df_results[df_results['anomaly_flag']]
            sns.scatterplot(data=anomalies, x=features[0], y=features[1], color='red',
                            marker='X', s=100, label='Anomaly')
            plt.title("Clusters and Anomalies")
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.legend()
            plt.show()
        else:
            print("Not enough features for scatter plot visualization.")

# =============================================================================
# STEP 4: Main Execution Function
# =============================================================================

def main():
    # ----- Define Configuration -----
    # Columns to drop (adjust these lists to your dataset)
    cols_to_drop = [
        'Account_ID', 'Customer_Account_Name', 'Customer_Legal_Identifier',
        'Trader', 'Instrument_Name', 'Execution_Venue'
    ]

    categorical_cols = [
        'Direction', 'Transaction_Status', 'Match_Status', 'Confirmation_Status',
        'DTC_Eligible', 'Depository', 'Place_Of_Settlement', 'Clearing_House_Source', 'Broker'
    ]

    numeric_cols = [
        'Quantity', 'Execution_Price', 'Principal_Amount', 'Net_Amount',
        'SEC_Fee', 'FINRA_FEE', 'Exchange_Fee', 'Fee_Difference'
    ]

    # ----- 1. Load the Data -----
    df = load_data_from_upload()

    # ----- 2. Initial Data Exploration -----
    print("\n--- Initial Data Exploration ---")
    initial_exploration(df)

    # ----- 3. Clean the Data -----
    df_clean = clean_data(df, cols_to_drop)
    print(f"\nAfter cleaning, data shape: {df_clean.shape}")

    # ----- 4. Encode Categorical Variables -----
    df_encoded = encode_categoricals(df_clean, categorical_cols)

    # ----- 5. Visualize Numeric Distributions -----
    print("\n--- Visualizing Numeric Distributions ---")
    visualize_numeric_distributions(df_encoded, numeric_cols)

    # ----- 6. Prepare Features for Modeling -----
    X, X_scaled, feature_list = prepare_features(df_encoded, numeric_cols, categorical_cols)
    print(f"\nFeatures used for modeling: {feature_list}")

    # ----- 7. Determine Optimal Number of Clusters -----
    print("\n--- Determining Optimal Number of Clusters ---")
    optimal_k, score_dict = find_optimal_k(X_scaled, k_min=2, k_max=10)

    # ----- 8. Anomaly Detection via Advanced KMeans Logic -----
    print("\n--- Anomaly Detection using Advanced KMeans ---")
    # Use n_mad=3; adjust if you wish to be more or less sensitive
    df_results = detect_anomalies_kmeans_advanced(X_scaled, df_encoded, optimal_k, n_mad=3)

    # ----- 9. Review Top Anomalies -----
    anomalies_sorted = df_results.sort_values('distance_to_center', ascending=False)
    print("\nTop 10 anomalies based on distance to cluster center (advanced method):")
    print(anomalies_sorted.head(10))

    # ----- 10. Visualize Clusters and Anomalies -----
    visualize_clusters(df_results, feature_list, use_pca=True)

    # Optionally: Save the results to a CSV file
    output_filename = "anomaly_detection_results.csv"
    df_results.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")

# =============================================================================
# STEP 5: Run the Pipeline
# =============================================================================

if __name__ == "__main__":
    main()
