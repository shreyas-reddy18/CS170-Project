"""
CS170 Project 2 - Part II
Nearest Neighbor Classifier and Leave-One-Out Validator

Authors: Shreyas, Rishabh, Rehan
"""
import math
from typing import Tuple, List, Sequence

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

class NearestNeighborClassifier:
    """
    Simple 1-Nearest Neighbor classifier using Euclidean distance.

    Train:
        stores the training instances.

    Test:
        given a test instance, returns the predicted class label
        of the closest training instance (under Euclidean distance)
        restricted to a given feature subset.
    """

    def __init__(self, feature_subset: Sequence[int]):
        # Feature indices are 1-based (1..N) in the project spec
        # Internally we keep them as 0-based indices into the feature vector
        self.feature_indices = [f - 1 for f in feature_subset]
        self.train_features: List[List[float]] = []
        self.train_labels: List[float] = []

    def train(self, training_data: List[List[float]]) -> None:
        """
        Store training data.

        Args:
            training_data: list of rows [class_label, f1, ..., fN]
        """
        self.train_labels = [row[0] for row in training_data]
        self.train_features = [row[1:] for row in training_data]

    def test(self, instance: List[float]) -> float:
        """
        Predict the class label for a single test instance.

        Args:
            instance: [class_label, f1, ..., fN] 

        Returns:
            predicted class label (float, 1.0 or 2.0)
        """
        if not self.train_features:
            raise RuntimeError("Classifier has not been trained")

        test_features = instance[1:]
        best_dist = float("inf")
        best_label = None

        for label, train_vec in zip(self.train_labels, self.train_features):
            dist = 0.0
            for idx in self.feature_indices:
                diff = test_features[idx] - train_vec[idx]
                dist += diff * diff
            # No need to take sqrt for comparison
            if dist < best_dist:
                best_dist = dist
                best_label = label

        return best_label


class LeaveOneOutValidator:
    """
    Leave-One-Out Cross-Validation for a given classifier and feature subset.
    """

    def __init__(self, dataset: List[List[float]]):
        self.dataset = dataset