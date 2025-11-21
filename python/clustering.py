import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import os
import sys

# Add current directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_and_preprocess_data

def run_clustering_analysis(filepath):
    # Load data
    X, labels_true, df_original = load_and_preprocess_data(filepath)
    
    # Convert labels_true (Not_Canceled/Canceled) to numeric for NMI/ARI
    # We don't use this for training, only for evaluation
    labels_true_numeric = labels_true.map({'Not_Canceled': 0, 'Canceled': 1})
    if labels_true_numeric.isnull().any():
        # Handle cases where status might be different strings
        print("Warning: Unexpected values in booking.status. Mapping to codes.")
        labels_true_numeric = labels_true.astype('category').cat.codes

    # --- Preview: show first 10 rows for clarity (original + preprocessed) ---
    print("\n--- Preview: First 10 rows (original data) ---")
    try:
        print(df_original.head(10).to_string(index=False))
    except Exception:
        print(df_original.head(10))

    print("\n--- Preview: First 10 rows (preprocessed features) ---")
    try:
        print(X.head(10).to_string(index=False))
    except Exception:
        print(X.head(10))

    print("\n--- Step 5: Clustering Analysis ---")
    print("We will test K (number of clusters) from 2 to 10.")
    
    k_range = range(2, 11)
    kmeans_inertias = []
    kmeans_silhouette_scores = []
    kmeans_nmi_scores = []
    kmeans_ari_scores = []
    
    hierarchical_silhouette_scores = []
    hierarchical_nmi_scores = []
    hierarchical_ari_scores = []

    print("\nRunning K-Means and Hierarchical Clustering for each K...")
    print(f"{'K':<5} | {'Algo':<15} | {'Inertia':<10} | {'Silhouette':<10} | {'NMI':<10} | {'ARI':<10}")
    print("-" * 75)

    for k in k_range:
        # --- K-Means ---
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X, kmeans_labels)
        nmi = normalized_mutual_info_score(labels_true_numeric, kmeans_labels)
        ari = adjusted_rand_score(labels_true_numeric, kmeans_labels)
        
        kmeans_inertias.append(inertia)
        kmeans_silhouette_scores.append(sil_score)
        kmeans_nmi_scores.append(nmi)
        kmeans_ari_scores.append(ari)
        
        print(f"{k:<5} | {'K-Means':<15} | {inertia:<10.2f} | {sil_score:<10.4f} | {nmi:<10.4f} | {ari:<10.4f}")

        # --- Hierarchical ---
        # Using Ward linkage which minimizes variance (similar to K-Means)
        hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        hc_labels = hc.fit_predict(X)
        
        sil_score_hc = silhouette_score(X, hc_labels)
        nmi_hc = normalized_mutual_info_score(labels_true_numeric, hc_labels)
        ari_hc = adjusted_rand_score(labels_true_numeric, hc_labels)
        
        hierarchical_silhouette_scores.append(sil_score_hc)
        hierarchical_nmi_scores.append(nmi_hc)
        hierarchical_ari_scores.append(ari_hc)
        
        print(f"{k:<5} | {'Hierarchical':<15} | {'-':<10} | {sil_score_hc:<10.4f} | {nmi_hc:<10.4f} | {ari_hc:<10.4f}")
        print("-" * 75)

    print("\n--- Step 6: Visualization ---")
    print("Generating plots for Elbow Method and Silhouette Scores...")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow Plot (K-Means only)
    axes[0].plot(k_range, kmeans_inertias, marker='o', linestyle='-', color='b')
    axes[0].set_title('Elbow Method (K-Means)')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia (Sum of Squared Distances)')
    axes[0].grid(True)
    
    # Silhouette Plot (Comparison)
    axes[1].plot(k_range, kmeans_silhouette_scores, marker='o', linestyle='-', label='K-Means', color='b')
    axes[1].plot(k_range, hierarchical_silhouette_scores, marker='s', linestyle='--', label='Hierarchical', color='r')
    axes[1].set_title('Silhouette Score Comparison')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(filepath), 'clustering_metrics.png')
    plt.savefig(plot_path)
    print(f"Plots saved to: {plot_path}")
    
    print("\n--- Analysis Complete ---")
    print("Review the table above and the generated plot to select the optimal number of clusters.")
    print("Higher Silhouette Score indicates better defined clusters.")
    print("For Elbow Method, look for the 'knee' where inertia decrease slows down.")
    print("NMI and ARI show how well the clusters align with the original 'Canceled/Not Canceled' status (Ground Truth).")

if __name__ == "__main__":
    # Path to the CSV file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'project_cluster.csv')
    
    if os.path.exists(csv_path):
        run_clustering_analysis(csv_path)
    else:
        print(f"Error: Data file not found at {csv_path}")
