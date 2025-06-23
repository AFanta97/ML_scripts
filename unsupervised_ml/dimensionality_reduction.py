#!/usr/bin/env python3

"""
dimensionality_reduction.py

This script performs dimensionality reduction (PCA or KernelPCA with RBF) on a dataset provided as a CSV file.
It supports:
- Automatic rescaling using StandardScaler or MinMaxScaler
- Visual inspection via correlation heatmap and scree plot
- Component retention based on a desired variance threshold
- Exporting reduced datasets
"""

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import os

# -------------------- Argument Parsing -------------------- #
parser = argparse.ArgumentParser(description="Dimensionality Reduction using PCA or KernelPCA")

parser.add_argument('--input', required=True, help='Input CSV file')
parser.add_argument('--output', default=None, help='Output filename for reduced dataset')
parser.add_argument('--method', choices=['pca', 'kernelpca'], default='pca', help='Reduction method to use')
parser.add_argument('--threshold', type=float, default=0.9, help='Information retention threshold (e.g., 0.9 for 90%%)')
parser.add_argument('--keep_all', action='store_true', help='Keep all reduced components instead of threshold-based reduction')
parser.add_argument('--scaler', choices=['standard', 'minmax'], default='standard', help='Scaler to apply to the data')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma parameter for RBF Kernel PCA')
parser.add_argument('--show_scree', action='store_true', help='Show explained variance ratio plot')
parser.add_argument('--save_dir', default='.', help='Directory where outputs (plots, CSV) will be saved')

args = parser.parse_args()

# -------------------- Load Dataset -------------------- #
if not os.path.isfile(args.input):
    raise FileNotFoundError(f"Input file {args.input} does not exist.")

df = pd.read_csv(args.input)
if df.isnull().any().any():
    raise ValueError("Dataset contains missing values. Please clean your data before running dimensionality reduction.")

# -------------------- Rescaling -------------------- #
scaler = StandardScaler() if args.scaler == 'standard' else MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# -------------------- Correlation Heatmap -------------------- #
plt.figure(figsize=(10, 8))
sns.heatmap(scaled_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
heatmap_path = os.path.join(args.save_dir, 'correlation_heatmap.png')
heatmap_svg_path = os.path.join(args.save_dir, 'correlation_heatmap.svg')
plt.savefig(heatmap_path)
plt.savefig(heatmap_svg_path)
plt.close()

# -------------------- Dimensionality Reduction -------------------- #
if args.method == 'pca':
    reducer = PCA()
    reducer.fit(scaled_data)
    explained_variance = reducer.explained_variance_ratio_

    # Scree plot
    if args.show_scree:
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(explained_variance), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot')
        scree_path = os.path.join(args.save_dir, 'scree_plot.png')
        scree_svg_path = os.path.join(args.save_dir, 'scree_plot.svg')
        plt.savefig(scree_path)
        plt.savefig(scree_svg_path)
        plt.show()

    # Determine how many components to retain
    if args.keep_all:
        n_components = len(df.columns)
    else:
        cumulative_variance = np.cumsum(explained_variance)
        n_components = np.searchsorted(cumulative_variance, min(args.threshold, 1.0)) + 1
        print(f"Retaining {n_components} components to preserve {args.threshold * 100:.1f}% variance.")

    pca_final = PCA(n_components=n_components)
    reduced_data = pca_final.fit_transform(scaled_data)

elif args.method == 'kernelpca':
    if args.keep_all:
        n_components = len(df.columns)
    else:
        n_components = int(len(df.columns) * args.threshold)
        n_components = min(n_components, len(df.columns))

    print(f"KernelPCA: Reducing to {n_components} components using RBF kernel (gamma={args.gamma})")
    reducer = KernelPCA(n_components=n_components, kernel='rbf', gamma=args.gamma)
    reduced_data = reducer.fit_transform(scaled_data)

# -------------------- Save Reduced Dataset -------------------- #
reduced_df = pd.DataFrame(reduced_data)
basename = os.path.splitext(os.path.basename(args.input))[0]
output_filename = args.output or f"{basename}_reduced.csv"
output_path = os.path.join(args.save_dir, output_filename)
reduced_df.to_csv(output_path, index=False)
print(f"Reduced dataset saved to {output_path}")
