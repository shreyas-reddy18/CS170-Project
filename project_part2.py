"""
CS170 Project 2 - Part II
Nearest Neighbor Classifier and Leave-One-Out Validator

Authors: Shreyas, Rishabh, Rehan
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
    global validator

    print("Welcome to Team Blue's Feature Selection Algorithm.")

    dataset_path = input("\nType in the name of the file to test: ").strip()

    try:
        dataset, num_features = load_dataset(dataset_path)
    except FileNotFoundError:
        print(f"Error: Could not find file '{dataset_path}'. Please make sure it is in this directory.")
        return

    print(f"\nThis dataset has {num_features} features (not including the class attribute), with {len(dataset)} instances.")

    print("\nPlease wait while I normalize the data... ", end="")
    normalize_dataset(dataset)
    print("Done!\n")

   
    validator = LeaveOneOutValidator(dataset)

    print("Type the number of the algorithm you want to run.")
    print("  1) Forward Selection")
    print("  2) Backward Elimination")
    print("  3) Team Blue's Special Algorithm")

    choice = input("\n").strip()
    print()

    try:
        if choice == '1':
            forward_selection(num_features)
        elif choice == '2':
            backward_elimination(num_features)
        elif choice == '3':
            special_algorithm(num_features)
        else:
            print("Invalid selection. Please run the program again and choose 1, 2, or 3.")

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    main()
