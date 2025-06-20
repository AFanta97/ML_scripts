import os
import argparse
import numpy as np
import pandas as pd


def load_data_to_dataframe(input_paths):
    """
    Load data from various formats into a pandas DataFrame.
    Supports: .txt, .npy, .csv, .xlsx
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    multiple_files = len(input_paths) > 1
    data_dict = {}

    for path in input_paths:
        filename = os.path.basename(path)
        feature_name = os.path.splitext(filename)[0]

        if path.endswith('.txt'):
            with open(path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # Header + Column
            if len(lines) > 1 and all(len(line.split()) == 1 for line in lines[1:]):
                header = lines[0]
                values = [float(x) for x in lines[1:]]
                data = np.array(values).reshape(-1, 1)
                col_names = [header]

            # Header + Row
            elif len(lines) == 2 and len(lines[0].split()) > 1:
                header = lines[0].split()
                values = [float(x) for x in lines[1].split()]
                data = np.array(values).reshape(1, -1)
                col_names = header

            # Column only
            elif len(lines) > 1 and all(len(line.split()) == 1 for line in lines):
                values = [float(x) for x in lines]
                data = np.array(values).reshape(-1, 1)
                col_names = [feature_name]

            # Row only
            elif len(lines) == 1:
                values = [float(x) for x in lines[0].split()]
                data = np.array(values).reshape(1, -1)
                col_names = [feature_name + f"_{i}" for i in range(data.shape[1])]

            # Multi-row/column
            else:
                try:
                    df = pd.read_csv(path, delim_whitespace=True)
                    return df
                except:
                    df = pd.read_csv(path, delim_whitespace=True, header=None)
                    df.columns = [f"feature_{i}" for i in range(df.shape[1])]
                    return df

        elif path.endswith('.npy'):
            raw = np.load(path)
            if raw.ndim == 1:
                data = raw.reshape(-1, 1)
                col_names = [feature_name]
            elif raw.ndim == 2:
                data = raw
                col_names = [f"{feature_name}_{i}" for i in range(data.shape[1])]
            else:
                raise ValueError(f"Unsupported shape for {filename}: {raw.shape}")

        elif path.endswith('.csv'):
            df = pd.read_csv(path)
            return df

        elif path.endswith('.xlsx'):
            df = pd.read_excel(path)
            return df

        else:
            raise ValueError(f"Unsupported file type: {path}")

        if multiple_files:
            if "length" in data_dict and data.shape[0] != data_dict["length"]:
                raise ValueError(f"Dimension mismatch in {filename}")
            data_dict[feature_name] = data.flatten()
            data_dict["length"] = data.shape[0]
        else:
            return pd.DataFrame(data, columns=col_names)

    data_dict.pop("length", None)
    return pd.DataFrame(data_dict)


def save_dataframe(df, output_path):
    """
    Save the dataframe to CSV, Excel, or NPY.
    """
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    elif output_path.endswith('.npy'):
        np.save(output_path, df.values)
    else:
        raise ValueError("Unsupported output format. Use .csv, .xlsx, or .npy")


def run_tests():
    """
    Simple test cases using synthetic data files.
    """
    print("Running test examples...\n")

    os.makedirs("test_data", exist_ok=True)

    # Create test .txt file with column and header
    with open("test_data/column_header.txt", "w") as f:
        f.write("feature1\n1\n2\n3\n4")

    # Create test .npy file with row
    np.save("test_data/row.npy", np.array([1, 2, 3, 4]))

    # Create test CSV
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv("test_data/example.csv", index=False)

    # Create test Excel
    pd.DataFrame({"x": [10, 20, 30]}).to_excel("test_data/example.xlsx", index=False)

    test_files = [
        "test_data/column_header.txt",
        "test_data/row.npy",
        "test_data/example.csv",
        "test_data/example.xlsx"
    ]

    for path in test_files:
        print(f"Loading: {path}")
        df = load_data_to_dataframe(path)
        print(df, "\n")


def main():
    parser = argparse.ArgumentParser(description="Load data files into a pandas DataFrame.")
    parser.add_argument('--input', nargs='+', help="Input file(s): .txt, .npy, .csv, .xlsx", required=False)
    parser.add_argument('--output', help="Output file to save DataFrame (.csv, .xlsx, .npy)", required=False)
    parser.add_argument('--test', action='store_true', help="Run test examples")

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if not args.input:
        print("Error: You must provide at least one input file unless using --test")
        return

    df = load_data_to_dataframe(args.input)
    print("Loaded DataFrame:")
    print(df)

    if args.output:
        save_dataframe(df, args.output)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
