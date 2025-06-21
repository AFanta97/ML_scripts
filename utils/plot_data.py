import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Import the data loader from your utility module
from load_data_to_df import load_data_to_dataframe

def plot_data(df, output_path_base, save_svg=False, show_plot=False):
    # Sanity check: no missing values
    if df.isnull().any().any():
        raise ValueError("Data contains missing values. Please clean the data before plotting.")

    # Drop non-numeric columns with a warning
    non_numeric_cols = df.select_dtypes(exclude='number').columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Dropping non-numeric columns: {list(non_numeric_cols)}")
        df = df.select_dtypes(include='number')

    if df.shape[1] < 1:
        raise ValueError("No numeric columns available for plotting.")
    elif df.shape[1] == 1:
        feature = df.columns[0]
        fig, axs = plt.subplots(3, 1, figsize=(6, 10))
        sns.histplot(df[feature], ax=axs[0], kde=True)
        axs[0].set_title(f"Histogram of {feature}")
        sns.violinplot(y=df[feature], ax=axs[1])
        axs[1].set_title(f"Violin plot of {feature}")
        sns.boxplot(y=df[feature], ax=axs[2])
        axs[2].set_title(f"Boxplot of {feature}")
        plt.tight_layout()
    elif df.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], ax=ax)
        ax.set_title("Scatterplot")
        plt.tight_layout()
    else:
        if df.shape[1] > 10:
            print("⚠️  Warning: More than 10 features may make the pairplot slow or unreadable.")
        fig = sns.pairplot(df)
    
    # Save the plot(s)
    if isinstance(fig, sns.axisgrid.PairGrid):  # For pairplot
        fig.fig.savefig(f"{output_path_base}.png")
        if save_svg:
            fig.fig.savefig(f"{output_path_base}.svg")
        with open(f"{output_path_base}.fig.pickle", "wb") as f:
            pickle.dump(fig.fig, f)
    else:  # For matplotlib figures
        fig.savefig(f"{output_path_base}.png")
        if save_svg:
            fig.savefig(f"{output_path_base}.svg")
        with open(f"{output_path_base}.fig.pickle", "wb") as f:
            pickle.dump(fig, f)
    
    if show_plot:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot data from various formats using seaborn.")
    parser.add_argument("--input", nargs="+", required=True, help="Path to one or more data files (.txt, .npy, .csv, .xlsx)")
    parser.add_argument("--output", default="plot", help="Base output file path (without extension)")
    parser.add_argument("--svg", action="store_true", help="Also save the plot as SVG")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    parser.add_argument("--features", nargs="+", help="Optional list of features to include in the plot")

    args = parser.parse_args()

    # Load the data
    df = load_data_to_dataframe(args.input)

    # Optionally filter features
    if args.features:
        missing = [f for f in args.features if f not in df.columns]
        if missing:
            raise ValueError(f"The following requested features were not found in the data: {missing}")
        df = df[args.features]

    # Plot
    plot_data(df, args.output, save_svg=args.svg, show_plot=args.show)


if __name__ == "__main__":
    main()
