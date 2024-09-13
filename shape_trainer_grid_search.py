import os
import re
import argparse
import dlib
from itertools import product

# Parsing arguments
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, default='train.xml',
        help="Training data (default = train.xml)", metavar='')
    ap.add_argument("-t", "--test", type=str, default=None,
        help="Test data (default = None). If not provided, no testing is done", metavar='')
    ap.add_argument("-o", "--out", type=str, default='predictor',
        help="Output filename (default = predictor)", metavar='')
    ap.add_argument("-th", "--threads", type=int, default=1,
        help="Number of threads to be used (default = 1)", metavar='')
    ap.add_argument("-dp", "--tree-depth", type=int, default=4,
        help="Choice of tree depth (default = 4)", metavar='')
    ap.add_argument("-c", "--cascade-depth", type=int, default=15,
        help="Choice of cascade depth (default = 15)", metavar='')
    ap.add_argument("-nu", "--nu", type=float, default=0.1,
        help="Regularization parameter (default = 0.1)", metavar='')
    ap.add_argument("-os", "--oversampling", type=int, default=10,
        help="Oversampling amount (default = 10)", metavar='')
    ap.add_argument("-s", "--test-splits", type=int, default=20,
        help="Number of test splits (default = 20)", metavar='')
    ap.add_argument("-f", "--feature-pool-size", type=int, default=500,
        help="Choice of feature pool size (default = 500)", metavar='')
    ap.add_argument("-n", "--num-trees", type=int, default=500,
        help="Number of regression trees (default = 500)", metavar='')
    return vars(ap.parse_args())

# Training and evaluation functions
def train_and_evaluate(train_path, test_path, output_path, options):
    try:
        # Train the model
        dlib.train_shape_predictor(train_path, output_path, options)
        training_error = dlib.test_shape_predictor(train_path, output_path)
        print(f"Training error (average pixel deviation): {training_error}")

        # Test the model if test data is provided
        if test_path:
            testing_error = dlib.test_shape_predictor(test_path, output_path)
            print(f"Testing error (average pixel deviation): {testing_error}")
        else:
            testing_error = None

        return training_error, testing_error

    except Exception as e:
        print(f"Error during training: {e}")
        return None, None

def extract_metrics(training_error, testing_error):
    metrics = {
        'training_error': training_error,
        'testing_error': testing_error
    }
    return metrics

def run_training(params, dataset, test_data, output_dir):
    # Set up training options
    options = dlib.shape_predictor_training_options()
    options.num_trees_per_cascade_level = params['num_trees']
    options.nu = params['nu']
    options.num_threads = params['threads']
    options.tree_depth = params['tree_depth']
    options.cascade_depth = params['cascade_depth']
    options.feature_pool_size = params['feature_pool_size']
    options.num_test_splits = params['test_splits']  # Provide a default value
    options.oversampling_amount = params['oversampling']
    options.be_verbose = True

    # Prepare output path
    output_path = os.path.join(output_dir, f"predictor_th{params['threads']}_dp{params['tree_depth']}_c{params['cascade_depth']}_nu{params['nu']}_os{params['oversampling']}_f{params['feature_pool_size']}_n{params['num_trees']}.dat")

    # Train and evaluate
    training_error, testing_error = train_and_evaluate(dataset, test_data, output_path, options)
    metrics = extract_metrics(training_error, testing_error)

    # Save metrics to CSV file
    metrics_file_path = os.path.join(output_dir, 'performance_metrics.csv')
    with open(metrics_file_path, 'a') as metrics_file:
        if os.stat(metrics_file_path).st_size == 0:
            metrics_file.write('threads,tree_depth,cascade_depth,nu,oversampling,feature_pool_size,num_trees,training_error,testing_error\n')
        metrics_file.write(f"{params['threads']},{params['tree_depth']},{params['cascade_depth']},{params['nu']},{params['oversampling']},{params['feature_pool_size']},{params['num_trees']},{metrics['training_error']},{metrics['testing_error']}\n")

def grid_search(param_grid, dataset, test_data, output_dir):
    keys, values = zip(*param_grid.items())
    for combination in product(*values):
        params = dict(zip(keys, combination))
        print(f"Running training with parameters: {params}")
        run_training(params, dataset, test_data, output_dir)

def main():
    args = parse_args()

    # Define parameter grid directly in the script
    param_grid = {
        'test_splits': [5, 10, 15],
        'threads': [7],
        'tree_depth': [2, 4, 6],
        'cascade_depth': [5, 10, 15],
        'nu': [0.05, .2, .8],
        'oversampling': [5, 10, 15],
        'feature_pool_size': [150, 300, 500],
        'num_trees': [100]
    }

    # Create output directory if it doesn't exist
    output_dir = 'training_results'
    os.makedirs(output_dir, exist_ok=True)

    # Perform grid search
    grid_search(param_grid, args['dataset'], args['test'], output_dir)

if __name__ == "__main__":
    main()