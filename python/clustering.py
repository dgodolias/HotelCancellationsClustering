import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
from matplotlib.backends.backend_pdf import PdfPages

# Configuration
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style('whitegrid')

# Ensure visuals directory exists inside python folder
VISUALS_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
if not os.path.exists(VISUALS_DIR):
    os.makedirs(VISUALS_DIR)

def save_plot(filename):
    """Saves the current plot to the visuals directory and closes it."""
    path = os.path.join(VISUALS_DIR, filename)
    plt.savefig(path)
    plt.close()

def fix_invalid_date(date_str):
    """Tries to fix invalid dates by decreasing the day by 1."""
    try:
        return pd.to_datetime(date_str)
    except:
        try:
            date_str = str(date_str)
            if '/' in date_str:
                parts = date_str.split('/')
                month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            elif '-' in date_str:
                parts = date_str.split('-')
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                return pd.NaT
            
            while day > 0:
                try:
                    return pd.Timestamp(year=year, month=month, day=day)
                except:
                    day -= 1
        except:
            pass
        return pd.NaT

def main():
    print("Loading dataset...")
    try:
        # Load dataset relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, '../project cluster.xlsx')
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please check the file path.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    
    # --- Preprocessing ---
    print("Preprocessing data...")
    df_processed = df.copy()
    
    # Feature Engineering
    df_processed['Total_Guests'] = df_processed['number.of.adults'] + df_processed['number.of.children']
    df_processed['Is_Family'] = (df_processed['number.of.children'] > 0).astype(int)
    df_processed['Total_Nights'] = df_processed['number.of.weekend.nights'] + df_processed['number.of.week.nights']
    df_processed['Cancellation_Ratio'] = (df_processed['P.C'] + 1) / (df_processed['P.not.C'] + df_processed['P.C'] + 2)
    df_processed['Price_per_Person'] = df_processed['average.price'] / df_processed['Total_Guests']
    df_processed['Price_per_Person'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Date fixing - R script drops it, so we drop it too. 
    # (R: df <- df %>% select(-date.of.reservation))
    if 'date.of.reservation' in df_processed.columns:
        df_processed.drop(columns=['date.of.reservation'], inplace=True)
    
    # Value capping/binning - Matching R script logic
    df_processed['number.of.children'] = df_processed['number.of.children'].apply(lambda x: 0 if x == 0 else 1)
    df_processed['number.of.weekend.nights'] = df_processed['number.of.weekend.nights'].apply(lambda x: min(x, 3))
    df_processed['number.of.week.nights'] = df_processed['number.of.week.nights'].apply(lambda x: min(x, 6))
    df_processed['P.C'] = df_processed['P.C'].apply(lambda x: 0 if x == 0 else 1)
    df_processed['P.not.C'] = df_processed['P.not.C'].apply(lambda x: 0 if x == 0 else 1)
    df_processed['Total_Guests'] = df_processed['Total_Guests'].apply(lambda x: 1 if x == 0 else min(x, 3))
    df_processed['Total_Nights'] = df_processed['Total_Nights'].round().astype(int).apply(lambda x: min(x, 8))
    
    # Save booking status and drop unused columns
    booking_status = df_processed['booking.status'].copy()
    df_processed.drop(columns=['Booking_ID', 'booking.status'], inplace=True)
    
    # One-hot encoding
    categorical_features = ['market.segment.type']
    # drop_first=True will drop the first alphabetical category. 
    # If 'Aviation' is present and first, it will be dropped, matching R's logic.
    df_encoded = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)
    
    # Drop meal, room type, week related columns
    cols_to_drop = [col for col in df_encoded.columns if col.startswith(('type.of.meal', 'room.type'))] 
    df_encoded.drop(columns=cols_to_drop, inplace=True)
    
    # Fill NaNs
    if df_encoded.isnull().sum().sum() > 0:
        df_encoded = df_encoded.fillna(df_encoded.mean())
        
    # Scaling (0 to 1) - Changed from (-1, 1) to match R
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled_array = scaler.fit_transform(df_encoded)
    df_scaled = pd.DataFrame(df_scaled_array, columns=df_encoded.columns)
    
    print("Data processed and scaled.")
    
    # --- Plotting Distributions ---
    print("Generating distribution plots...")
    numeric_cols = [
        'Price_per_Person', 'Cancellation_Ratio', 'Total_Nights', 'Is_Family', 
        'Total_Guests', 'special.requests', 'average.price', 'P.not.C', 'P.C', 
        'repeated', 'lead.time', 'car.parking.space', 'number.of.week.nights', 
        'number.of.weekend.nights', 'number.of.children', 'number.of.adults'
    ]
    
    for col in numeric_cols:
        if col in df_scaled.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df_scaled[col], bins=30, edgecolor='black', alpha=0.7)
            plt.title(f'Distribution (Scaled): {col}')
            plt.xlabel('Scaled Value [0, 1]')
            plt.ylabel('Frequency')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_plot(f'dist_{col}.png')

    # --- Clustering ---
    print("Running clustering algorithms...")
    X = df_scaled.values
    k_range = range(2, 11)
    
    kmeans_silhouette = []
    hierarchical_silhouette = []
    nmi_scores = []
    ari_scores = []
    
    kmeans_labels_dict = {}
    hierarchical_labels_dict = {}
    kmeans_inertia = []

    # 1. K-Means Loop
    print("\nΕκτέλεση K-Means για k = 2 έως 10...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        k_labels = kmeans.fit_predict(X)
        kmeans_labels_dict[k] = k_labels
        kmeans_inertia.append(kmeans.inertia_)
        s_score = silhouette_score(X, k_labels)
        kmeans_silhouette.append(s_score)
        print(f"K-Means με k={k}... ✓ (Inertia: {kmeans.inertia_:.2f}, Silhouette: {s_score:.4f})")
    print("K-Means ολοκληρώθηκε!")

    # 2. Hierarchical Loop
    print("\nΕκτέλεση Hierarchical Clustering για k = 2 έως 10...")
    
    # Dendrogram (using a sample for visibility)
    print("Generating Dendrogram...")
    plt.figure(figsize=(12, 8))
    # Use a sample for dendrogram to avoid overcrowding
    sample_idx = np.random.choice(X.shape[0], min(5000, X.shape[0]), replace=False)
    X_sample = X[sample_idx]
    Z = linkage(X_sample, method='ward')
    dendrogram(Z, truncate_mode='lastp', p=50, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage, Sample=5000)')
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')
    plt.tight_layout()
    save_plot('dendrogram.png')
    print("Dendrogram saved.")

    for k in k_range:
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
        h_labels = hierarchical.fit_predict(X)
        hierarchical_labels_dict[k] = h_labels
        s_score = silhouette_score(X, h_labels)
        hierarchical_silhouette.append(s_score)
        print(f"Hierarchical με k={k}... ✓ (Silhouette: {s_score:.4f})")
    print("Hierarchical Clustering ολοκληρώθηκε!")

    # 3. Comparison Loop
    print("\nΥπολογισμός NMI και ARI μεταξύ K-Means και Hierarchical...")
    for k in k_range:
        k_labels = kmeans_labels_dict[k]
        h_labels = hierarchical_labels_dict[k]
        nmi = normalized_mutual_info_score(k_labels, h_labels)
        ari = adjusted_rand_score(k_labels, h_labels)
        nmi_scores.append(nmi)
        ari_scores.append(ari)
        print(f"k={k}: NMI={nmi:.4f}, ARI={ari:.4f}")
    print("Υπολογισμοί ολοκληρώθηκαν!")
        
    # Evaluation Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    k_values = list(k_range)
    
    axes[0, 0].plot(k_values, kmeans_inertia, 'o-')
    axes[0, 0].set_title('K-Means Inertia')
    
    axes[0, 1].plot(k_values, kmeans_silhouette, 'o-', label='K-Means')
    axes[0, 1].plot(k_values, hierarchical_silhouette, 's-', label='Hierarchical')
    axes[0, 1].legend()
    axes[0, 1].set_title('Silhouette Score')
    
    axes[1, 0].plot(k_values, nmi_scores, 'o-')
    axes[1, 0].set_title('NMI Score')
    
    axes[1, 1].plot(k_values, ari_scores, 'o-')
    axes[1, 1].set_title('ARI Score')
    
    plt.tight_layout()
    save_plot('clustering_evaluation.png')
    
    # --- Feature Elimination (k=4) ---
    print("Running feature elimination analysis (k=4)...")
    k = 4
    baseline_silhouette = silhouette_score(X, kmeans_labels_dict[4])
    
    results = []
    for i, feature in enumerate(df_scaled.columns):
        X_reduced = np.delete(X, i, axis=1)
        kmeans_reduced = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_reduced = kmeans_reduced.fit_predict(X_reduced)
        score = silhouette_score(X_reduced, labels_reduced)
        results.append({'feature': feature, 'improvement': score - baseline_silhouette})
    
    sorted_results = sorted(results, key=lambda x: x['improvement'], reverse=True)
    print("Top 5 features to remove (improve silhouette):")
    for res in sorted_results[:5]:
        print(f"  {res['feature']}: +{res['improvement']:.4f}")

    # --- Final Clustering (k=4) ---
    print("Finalizing clusters (k=4)...")
    k_optimal = 4
    kmeans_optimal = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    cluster_labels = kmeans_optimal.fit_predict(X)
    
    # PCA Visualization
    print("Generating PCA plot...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
    plt.title(f'K-Means Clustering (k={k_optimal}) - PCA Projection')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    save_plot('pca_clusters.png')
    print("PCA plot saved.")
    
    df_with_clusters = df_encoded.copy()
    df_with_clusters['Cluster'] = cluster_labels
    df_with_clusters['booking.status'] = booking_status.values
    
    # Cluster Analysis Plots
    variables_to_analyze = [
        'number.of.adults', 'number.of.children', 'car.parking.space', 
        'lead.time', 'repeated', 'P.C', 'P.not.C', 'average.price', 
        'special.requests', 'Total_Guests', 'Is_Family', 'Total_Nights', 
        'Cancellation_Ratio', 'Price_per_Person', 'number.of.week.nights', 
        'number.of.weekend.nights'
    ]
    market_segment_cols = [c for c in df_with_clusters.columns if 'market.segment.type' in c]
    
    cluster_means = df_with_clusters.groupby('Cluster')[variables_to_analyze + market_segment_cols].mean()
    
    # Bar plots for variables
    for var in variables_to_analyze + market_segment_cols:
        plt.figure(figsize=(10, 6))
        plt.bar(cluster_means.index, cluster_means[var], color=['steelblue', 'coral', 'lightgreen', 'gold'])
        plt.title(f'Cluster Comparison: {var}')
        save_plot(f'cluster_comp_{var}.png')
        
    # Summary Table
    cluster_summary = cluster_means.copy()
    cluster_summary['Count'] = df_with_clusters.groupby('Cluster').size()
    cols_order = ['Count'] + variables_to_analyze + market_segment_cols
    cluster_summary = cluster_summary[cols_order]
    
    print("\nCluster Summary (Means):")
    print(cluster_summary)

    
    # Heatmap
    plt.figure(figsize=(16, 6))
    scaler_heatmap = MinMaxScaler()
    cluster_summary_norm = pd.DataFrame(
        scaler_heatmap.fit_transform(cluster_summary),
        columns=cluster_summary.columns,
        index=cluster_summary.index
    )
    sns.heatmap(cluster_summary_norm.T, annot=cluster_summary_norm.T.round(2), cmap='RdYlGn')
    plt.title('Cluster Means Heatmap (Normalized)')
    plt.tight_layout()
    save_plot('cluster_heatmap.png')
    
    # Cancellation Rate Plot
    cancellation_rates = df_with_clusters.groupby('Cluster')['booking.status'].apply(
        lambda x: (x == 'Canceled').mean() * 100
    )
    
    plt.figure(figsize=(10, 6))
    plt.bar(cancellation_rates.index, cancellation_rates, color=['steelblue', 'coral', 'lightgreen', 'gold'])
    plt.title('Cancellation Rate by Cluster')
    plt.ylabel('Cancellation Rate (%)')
    save_plot('cancellation_rate.png')
    
    print("\nCancellation Rates:")
    print(cancellation_rates)

    # Create PDF with all plots
    print("Creating PDF report...")
    # Save PDF in the python folder (script directory)
    pdf_path = os.path.join(os.path.dirname(__file__), 'clustering_report.pdf')
    
    # Get all png files in visuals dir
    plot_files = [f for f in os.listdir(VISUALS_DIR) if f.endswith('.png')]
    
    # Sort them to have a logical order: distributions first, then others
    plot_files.sort(key=lambda x: (not x.startswith('dist_'), x))

    with PdfPages(pdf_path) as pdf:
        for filename in plot_files:
            filepath = os.path.join(VISUALS_DIR, filename)
            # Read image
            img = plt.imread(filepath)
            
            # Create a figure to display the image
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off') # Hide axes
            plt.title(filename, fontsize=10)
            
            # Save to PDF
            pdf.savefig()
            plt.close()
            
    print(f"PDF report saved to: {pdf_path}")

if __name__ == "__main__":
    main()
