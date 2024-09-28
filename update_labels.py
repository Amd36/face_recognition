"""
Label Update Script for Custom Datasets

This script automatically generates a `labels.txt` file from the directory structure of a custom dataset.
Each subdirectory in the specified dataset directory corresponds to a label, and the names of these subdirectories are written to `labels.txt`. This is useful when working with external datasets that are organized by class labels in separate folders.

### Features:
1. **Automatic Label Extraction**: Extracts label names from the subdirectory names in the dataset directory.
2. **Customizable Dataset Directory**: The dataset directory can be specified using a command-line argument.
3. **File Output**: Generates or updates the `labels.txt` file with the extracted label names.

### Command-line Arguments:
- `--dataset_dir`: Base directory of the external dataset (default is `datasets`).

### Output:
- `labels.txt`: A text file containing one label per line, based on the folder names in the dataset directory.

### Dependencies:
- os
- argparse

"""

import os
import argparse

def update_labels(dataset_dir):
    """Extract labels from the dataset directory and save them to labels.txt."""
    labels = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    with open("labels.txt", 'w') as file:
        for label in labels:
            file.write(f"{label}\n")


if __name__ == '__main__':
    # Command-line argument parser setup
    parser = argparse.ArgumentParser(description='Update the labels.txt with proper labels if external dataset is used')
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='Base directory of the external dataset')
    args = parser.parse_args()

    # Update labels from the specified dataset directory
    update_labels(args.dataset_dir)

    print("labels.txt updated!")
