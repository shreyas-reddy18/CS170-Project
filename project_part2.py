"""
CS170 Project 2 - Part II
Nearest Neighbor Classifier and Leave-One-Out Validator

Group: Shreyas Nallapareddy - snall008, Rishabh Pillai - rpill005, Rehan ALam - malam041

DatasetID:
    - Small Dataset: small-test-dataset-2-2.txt
    - Large Dataset: large-test-dataset-2.txt
    - Titanic Dataset: titanic clean-2.txt
    
Results:
- Small Dataset Results:
    - Forward: The best feature subset is {3,5}, which has an accuracy of 92.0%
    - Backward: The best feature subset is {2,4,5,7,10}, which has an accuracy of 82.0%
    - Special Algorithm: Best subset found: {5}, accuracy = 75.0%
- Large Dataset Results:
    - Forward: The best feature subset is {1,27}, which has an accuracy of 95.5%
    - Backward: The best feature subset is {2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40}, which has an accuracy of 72.2%
    - Special Algorithm: Best subset found: {1,27}, accuracy = 95.5%
- Titanic Dataset Results:
    - Foward: The best feature subset is {2}, which has an accuracy of 78.0%
    - Backward: The best feature subset is {2}, which has an accuracy of 78.0%
    - Special Algorithm: Best subset found: {2}, accuracy = 78.0%
"""
import math
import time
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

    def evaluate(self, feature_subset: Sequence[int], verbose: bool = False) -> float:
        """
        Perform Leave One Out Validation.

        Args:
            feature_subset: list/tuple of feature indices (1-based)
            verbose: if True, prints per-instance predictions (trace)

        Returns:
            accuracy as a float in [0,1]
        """
        n = len(self.dataset)
        if n == 0:
            raise ValueError("Empty dataset")

        correct = 0
        start_time = time.perf_counter()

        for i in range(n):
            # Split into training and test
            test_instance = self.dataset[i]
            train_data = [self.dataset[j] for j in range(n) if j != i]

            # Create and train classifier
            clf = NearestNeighborClassifier(feature_subset)
            clf.train(train_data)

            # Predict
            predicted = clf.test(test_instance)
            actual = test_instance[0]

            if predicted == actual:
                correct += 1

            if verbose:
                print(
                    f"Instance {i+1}/{n}: actual={int(actual)}, predicted={int(predicted)}"
                )

        end_time = time.perf_counter()
        accuracy = correct / n

        print(
            f"Finished Leave One Out Validation for features {feature_subset} "
            f"-> accuracy = {accuracy:.4f} ({correct}/{n})"
        )
        print(f"Time elapsed: {end_time - start_time:.4f} seconds\n")
            
        return accuracy

def main():
    print("CS170 Project 2 - Part II: Nearest Neighbor and Validator")
    dataset_path = input("Enter path to dataset file: ").strip()
    dataset, num_features = load_dataset(dataset_path)
    print(f"Loaded dataset with {len(dataset)} instances and {num_features} features.")

    normalize_dataset(dataset)
    print("Features normalized (zero mean, unit variance).\n")

    subset_str = input(
        "Enter feature subset as space-separated indices (e.g., '3 5 7'): "
    ).strip()
    if not subset_str:
        print("No features specified. Exiting.")
        return

    feature_subset = [int(x) for x in subset_str.split()]
    print(f"Using feature subset: {feature_subset}\n")

    validator = LeaveOneOutValidator(dataset)
    # Set verbose=True if you want per-instance trace lines
    validator.evaluate(feature_subset, verbose=False)


if __name__ == "__main__":
    main()
