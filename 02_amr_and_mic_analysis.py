import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skbio.stats.distance import permanova
from skbio import DistanceMatrix
from scipy.spatial.distance import pdist, squareform

def plot_amr_heatmap(amr_data_path, output_name="amr_heatmap.png"):
    """Generates a heatmap of AMR gene presence/absence."""
    df = pd.read_csv(amr_data_path, index_col=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, cmap="YlGnBu", cbar_kws={'label': 'Presence (1) / Absence (0)'})
    plt.title("AMR Gene Profile Heatmap")
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()

def perform_pca(feature_df, metadata_df, target_col, output_name="pca_plot.png"):
    """Performs PCA and plots the first two principal components."""
    # Standardize features for PCA
    features_scaled = StandardScaler().fit_transform(feature_df)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(features_scaled)
    
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, metadata_df[[target_col]].reset_index(drop=True)], axis=1)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue=target_col, data=pca_df, palette='Set2', s=100)
    plt.title(f"PCA of Genomic Features by {target_col}")
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()

def run_permanova(feature_df, metadata_df, target_col):
    """Runs PERMANOVA to test for significant differences between groups."""
    # Calculate Jaccard distance matrix for binary gene presence/absence
    dist_matrix = squareform(pdist(feature_df, metric='jaccard'))
    dm = DistanceMatrix(dist_matrix, ids=feature_df.index)
    
    results = permanova(dm, metadata_df, column=target_col, permutations=999)
    print(f"\n--- PERMANOVA Results for {target_col} ---")
    print(results)
    return results

def plot_stacked_bars(mic_df, output_name="mic_stacked_bars.png"):
    """Creates a stacked bar chart for MIC distributions."""
    mic_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title("MIC Distribution Across Isolates")
    plt.ylabel("Proportion / Count")
    plt.legend(title="MIC Value", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()

