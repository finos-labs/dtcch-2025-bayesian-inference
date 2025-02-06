import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def visualize_clusters(df_results, features, use_pca=True):
    """
    Visualizes clusters and anomalies.
    If use_pca is True, projects the data to 2D using PCA and returns the figure.
    Otherwise, attempts to use the first two features.
    """
    if use_pca:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(df_results[features])
        df_results = df_results.copy()  # avoid modifying the original
        df_results['pca1'] = X_pca[:, 0]
        df_results['pca2'] = X_pca[:, 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_results, x='pca1', y='pca2', hue='cluster_label',
                        palette='viridis', alpha=0.6, ax=ax)
        anomalies = df_results[df_results['anomaly_flag']]
        sns.scatterplot(data=anomalies, x='pca1', y='pca2', color='red',
                        marker='X', s=100, label='Anomaly', ax=ax)
        ax.set_title("Clusters and Anomalies (PCA Projection)")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend()
        return fig
    else:
        if len(features) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_results, x=features[0], y=features[1], hue='cluster_label',
                            palette='viridis', alpha=0.6, ax=ax)
            anomalies = df_results[df_results['anomaly_flag']]
            sns.scatterplot(data=anomalies, x=features[0], y=features[1], color='red',
                            marker='X', s=100, label='Anomaly', ax=ax)
            ax.set_title("Clusters and Anomalies")
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.legend()
            return fig
        else:
            print("Not enough features for scatter plot visualization.")
            return None
