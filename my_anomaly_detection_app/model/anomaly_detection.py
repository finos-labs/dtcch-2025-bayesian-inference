import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_optimal_k(X_scaled, k_min=2, k_max=10):
    """
    Determines the optimal number of clusters using silhouette score.
    Returns the optimal k and a dictionary of scores.
    """
    scores = {}
    for k in range(k_min, k_max + 1):
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
    Clusters data using KMeans and computes Euclidean distances from each point
    to its cluster center. Points with distance greater than (median + n_mad * MAD)
    within their cluster are flagged as anomalies.
    Returns the original DataFrame with added columns:
      - cluster_label
      - distance_to_center
      - anomaly_flag
    """
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_
    
    distances = np.linalg.norm(X_scaled - centers[cluster_labels], axis=1)
    anomaly_flags = np.zeros(len(distances), dtype=bool)
    
    for cluster in range(optimal_k):
        mask = (cluster_labels == cluster)
        cluster_distances = distances[mask]
        median_val = np.median(cluster_distances)
        mad = np.median(np.abs(cluster_distances - median_val))
        threshold = median_val + n_mad * mad
        anomaly_flags[mask] = cluster_distances > threshold
        print(f"Cluster {cluster}: median={median_val:.4f}, MAD={mad:.4f}, threshold={threshold:.4f}, anomalies={np.sum(cluster_distances > threshold)}")
    
    print(f"\nTotal anomalies detected: {np.sum(anomaly_flags)} out of {len(distances)} samples")
    df_results = original_df.copy()
    df_results['cluster_label'] = cluster_labels
    df_results['distance_to_center'] = distances
    df_results['anomaly_flag'] = anomaly_flags
    return df_results
