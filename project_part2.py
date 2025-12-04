"""
CS170 Project 2 - Part II
Nearest Neighbor Classifier and Leave-One-Out Validator

Authors: Shreyas, Rishabh, Rehan
"""

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