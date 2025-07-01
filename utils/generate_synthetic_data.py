import os
import argparse
import numpy as np
import pandas as pd
from sympy import symbols, lambdify, sympify

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Data Generator")

    # Basic input
    parser.add_argument('--func', type=str, default="x**2 + y", 
                        help="Symbolic function, e.g., 'x**2 + y'")
    parser.add_argument('--vars', type=str, default="x,y", 
                        help="Comma-separated list of variables (x, x,y, or x,y,z)")
    parser.add_argument('--samples', type=int, default=1000, 
                        help="Number of samples to generate")
    parser.add_argument('--range', type=float, nargs=2, default=[-1, 1], 
                        help="Range of input values (min max) for each variable")
    parser.add_argument('--noise', type=float, default=0.0, 
                        help="Amplitude of uniform random noise to add to output")

    # Correlated data generation
    parser.add_argument('--correlated', action='store_true', 
                        help="Generate correlated data with y = f(x1,...,xn) + noise")

    # Output
    parser.add_argument('--output_format', type=str, choices=['txt_one', 'txt_multi', 'npy_one', 'npy_multi', 'csv'], 
                        default='csv', help="Output format")
    parser.add_argument('--outdir', type=str, default='output_data', 
                        help="Directory to save the generated files")

    return parser.parse_args()

def generate_inputs(n_samples, n_vars, var_range):
    """Generates uniform random data"""
    return np.random.uniform(var_range[0], var_range[1], size=(n_samples, n_vars))

def generate_outputs(inputs, func_str, var_names, noise_amplitude):
    """Evaluates the symbolic function with optional noise"""
    sym_vars = symbols(var_names)
    expr = sympify(func_str)
    func = lambdify(sym_vars, expr, modules='numpy')
    y = func(*[inputs[:, i] for i in range(len(sym_vars))])
    y += np.random.uniform(-noise_amplitude, noise_amplitude, size=len(y))
    return y

def generate_correlated_data(n_samples, noise_amplitude):
    """Generates moderately correlated features with a target y"""
    x = np.random.uniform(-1, 1, size=(n_samples, 5))
    weights = np.array([0.3, -0.2, 0.5, 0.1, -0.4])
    y = x @ weights + np.random.normal(0, noise_amplitude, size=n_samples)
    return x, y

def save_data(X, y, output_format, outdir):
    os.makedirs(outdir, exist_ok=True)

    if output_format == 'txt_one':
        np.savetxt(os.path.join(outdir, 'data.txt'), np.column_stack((X, y)))
    elif output_format == 'txt_multi':
        for i in range(X.shape[1]):
            np.savetxt(os.path.join(outdir, f'x{i+1}.txt'), X[:, i])
        np.savetxt(os.path.join(outdir, 'y.txt'), y)
    elif output_format == 'npy_one':
        np.save(os.path.join(outdir, 'data.npy'), np.column_stack((X, y)))
    elif output_format == 'npy_multi':
        for i in range(X.shape[1]):
            np.save(os.path.join(outdir, f'x{i+1}.npy'), X[:, i])
        np.save(os.path.join(outdir, 'y.npy'), y)
    elif output_format == 'csv':
        col_names = [f'x{i+1}' for i in range(X.shape[1])] + ['y']
        df = pd.DataFrame(np.column_stack((X, y)), columns=col_names)
        df.to_csv(os.path.join(outdir, 'data.csv'), index=False)

def main():
    args = parse_args()

    if args.correlated:
        # Correlated case
        X, y = generate_correlated_data(args.samples, args.noise)
    else:
        # Symbolic function case
        var_names = [v.strip() for v in args.vars.split(',')]
        if len(var_names) > 3:
            raise ValueError("Maximum of 3 variables (x, y, z) supported.")
        X = generate_inputs(args.samples, len(var_names), args.range)
        y = generate_outputs(X, args.func, var_names, args.noise)

    save_data(X, y, args.output_format, args.outdir)
    print(f"Data saved in {args.outdir}/ using format '{args.output_format}'")

if __name__ == "__main__":
    main()
