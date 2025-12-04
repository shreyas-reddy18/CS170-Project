"""
CS170 Project 2 - Part II
Nearest Neighbor Classifier and Leave-One-Out Validator

Authors: Shreyas, Rishabh, Rehan
"""
import math
from typing import Tuple, List

def load_dataset(path: str) -> Tuple[List[List[float]], int]:
    """
    Load dataset from ASCII text file.

    Returns:
        dataset: list of rows, each [class_label, f1, f2, ..., fN]
        num_features: number of features (N)
    """
    dataset: List[List[float]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            row = [float(x) for x in parts]
            dataset.append(row)

    if not dataset:
        raise ValueError("Empty dataset")

    num_features = len(dataset[0]) - 1
    return dataset, num_features

def normalize_dataset(dataset: List[List[float]]) -> None:
    """
    Normalize feature columns to zero mean and unit standard deviation.

    This is done column-wise over all instances.

    Args:
        dataset: modified in place. Each row is [class_label, f1, ..., fN]
    """
    if not dataset:
        return

    n_instances = len(dataset)
    n_features = len(dataset[0]) - 1

    # Compute mean for each feature
    means = [0.0] * n_features
    for row in dataset:
        for j in range(n_features):
            means[j] += row[1 + j]
    means = [m / n_instances for m in means]

    # Compute standard deviation for each feature
    variances = [0.0] * n_features
    for row in dataset:
        for j in range(n_features):
            diff = row[1 + j] - means[j]
            variances[j] += diff * diff
    stds = [math.sqrt(v / n_instances) for v in variances]

    # Normalize
    for row in dataset:
        for j in range(n_features):
            if stds[j] > 0:
                row[1 + j] = (row[1 + j] - means[j]) / stds[j]
            else:
                # Feature has zero variance; set to 0 for all instances
                row[1 + j] = 0.0