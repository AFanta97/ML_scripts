import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import os
import sys
import warnings

# ----------------------------- #
#         Argument Parser       #
# ----------------------------- #
parser = argparse.ArgumentParser(description="Clustering script for unsupervised learning using KMeans or DBSCAN")

parser.add_argument("--input", type=str, required=True, help="Path to CSV file containing data")
parser.add_argument("--method", type=str, choices=["kmeans", "dbscan"], required=True, help="Clustering algorithm to use")
parser.add_argument("--scaler", type=str, choices=["standard", "minmax"], default="standard", help="Scaling method")

# KMeans-specific
parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters for KMeans")
parser.add_argument("--max_clusters", type=int, help="Maximum clusters for KMeans auto-selection (uses silhouette and elbow)")

# DBSCAN-specific
parser.add_argument("--eps", type=float, help="Epsilon for DBSCAN")
parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for DBSCAN")

# Visualization and Output
parser.add_argument("--visualize", action="store_true", help="Enable 2D visualization (PCA-based)")
parser.add_argument("--save_svg", action="store_true", help="Save plot also as SVG")
parser.add_argument("--save_dir", type=str, default=".", help="Directory to save outputs")

args = parser.parse_args()

# ----------------------------- #
#           Load Data           #
# ----------------------------- #
try:
    df = pd.read_csv(args.input)
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

if df.isnull().any().any():
    print("Error: Input data contains missing values.")
    sys.exit(1)

# ----------------------------- #
#         Data Scaling          #
# ----------------------------- #
if args.scaler == "standard":
    scaler = StandardScaler()
elif args.scaler == "minmax":
    scaler = MinMaxScaler()
else:
    print("Unknown scaler. Exiting.")
    sys.exit(1)

X_scaled = scaler.fit_transform(df.values)

# ----------------------------- #
#      KMeans: Best Cluster     #
# ----------------------------- #
def select_best_kmeans(X, max_k):
    silhouette_scores = []
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
        inertias.append(km.inertia_)

    best_k = k_range[np.argmax(silhouette_scores)]
    
    # Plot silhouette and inertia curves
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(k_range, silhouette_scores, 'g-', label='Silhouette Score')
    ax2.plot(k_range, inertias, 'b--', label='Inertia')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Silhouette Score', color='g')
    ax2.set_ylabel('Inertia', color='b')
    plt.title("KMeans Cluster Selection")
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "kmeans_selection_plot.png"))
    fig.savefig(os.path.join(args.save_dir, "kmeans_selection_plot.fig"))
    if args.save_svg:
        plt.savefig(os.path.join(args.save_dir, "kmeans_selection_plot.svg"))
    plt.close()

    return best_k

# ----------------------------- #
#        Clustering Logic       #
# ----------------------------- #
if args.method == "kmeans":
    if args.max_clusters:
        print(f"Performing automatic KMeans selection up to {args.max_clusters} clusters...")
        args.n_clusters = select_best_kmeans(X_scaled, args.max_clusters)
        print(f"Best number of clusters found: {args.n_clusters}")
    
    if not args.n_clusters:
        print("Please specify either --n_clusters or --max_clusters for KMeans.")
        sys.exit(1)
    
    model = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto')
    labels = model.fit_predict(X_scaled)

elif args.method == "dbscan":
    if not args.eps:
        print("DBSCAN requires --eps to be set.")
        sys.exit(1)
    model = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    labels = model.fit_predict(X_scaled)

    # Optional: warn if lots of noise
    n_noise = list(labels).count(-1)
    if n_noise / len(labels) > 0.5:
        warnings.warn("More than 50% of data points classified as noise (-1 label) by DBSCAN.")

else:
    print("Unknown clustering method.")
    sys.exit(1)

# ----------------------------- #
#        Save Cluster Data      #
# ----------------------------- #
df_clustered = df.copy()
df_clustered['cluster'] = labels
df_clustered.to_csv(os.path.join(args.save_dir, "clustered_data.csv"), index=False)

# ----------------------------- #
#        Visualization (2D)     #
# ----------------------------- #
if args.visualize:
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    df_vis = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_vis["cluster"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_vis, x="PC1", y="PC2", hue="cluster", palette="Set2")
    plt.title(f"{args.method.upper()} Clustering Visualization (2D PCA)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "clusters_plot.png"))
    plt.savefig(os.path.join(args.save_dir, "clusters_plot.fig"))
    if args.save_svg:
        plt.savefig(os.path.join(args.save_dir, "clusters_plot.svg"))
    plt.close()

print("Clustering complete. Results saved.")
