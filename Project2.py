"""
CS170 Project 2: Part 1

Authors: Shreyas, Rishabh, Rehan
"""

import random
import copy 

def leave_one_out_cross_validation(feature_subset, data=None):
    """
    Returns random accuracy.
    In Part II, this will be replaced with actual k-NN classifier.
    
    Args:
        feature_subset: List of feature indices to evaluate
        data: Dataset (unused in Part I stub)
    
    Returns:
        Random accuracy percentage (float)
    """
    return round(random.uniform(0, 100), 1)

def print_feature_set(features):
    """
    Format feature list as {1,2,3} to match required trace output.
    
    Args:
        features: List of feature indices
    
    Returns:
        Formatted string like "{1,2,3}" or "{}" for empty
    """
    if not features:
        return "{}"
    sorted_features = sorted(features)
    return "{" + ",".join(str(f) for f in sorted_features) + "}"

def forward_selection(total_features):
    """
    Greedy Forward Selection Algorithm.
    
    Strategy:
    - Start with empty feature set
    - At each level, try adding each remaining feature
    - Add the feature that gives highest accuracy
    - Stop when accuracy decreases (greedy hill-climbing)
    
    Args:
        total_features: Total number of features available
    """
    current_features = []  # Start with empty set
    best_overall_accuracy = 0
    best_overall_features = []
    
    # Evaluate baseline (no features)
    initial_accuracy = leave_one_out_cross_validation(current_features)
    print(f'Using no features and "random" evaluation, I get an accuracy of {initial_accuracy}%')
    print("\nBeginning search.\n")
    
    best_overall_accuracy = initial_accuracy
    best_overall_features = copy.deepcopy(current_features)
    
    # Iterate through levels (adding one feature at a time)
    for level in range(total_features):
        best_feature_to_add = None
        best_accuracy_at_level = -1
        
        # Try adding each feature not currently in the set
        for feature_id in range(1, total_features + 1):
            if feature_id not in current_features:
                # Create test set with this feature added
                test_features = current_features + [feature_id]
                accuracy = leave_one_out_cross_validation(test_features)
                
                print(f"    Using feature(s) {print_feature_set(test_features)} accuracy is {accuracy}%")
                
                # Track best feature at this level
                if accuracy > best_accuracy_at_level:
                    best_accuracy_at_level = accuracy
                    best_feature_to_add = feature_id
        
        # Add the best feature found at this level
        if best_feature_to_add is not None:
            current_features.append(best_feature_to_add)
            print(f"\nFeature set {print_feature_set(current_features)} was best, accuracy is {best_accuracy_at_level}%\n")
            
            # Check if we improved overall (greedy stopping condition)
            if best_accuracy_at_level > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_at_level
                best_overall_features = copy.deepcopy(current_features)
            else:
                # Accuracy decreased, so stop searching (hill-climbing)
                print("(Warning, Accuracy has decreased!)\n")
                break
        else:
            break
    
    print(f"Finished search!! The best feature subset is {print_feature_set(best_overall_features)}, which has an accuracy of {best_overall_accuracy}%")

def backward_elimination(total_features):
    """
    Greedy Backward Elimination Algorithm.
    
    Strategy:
    - Start with all features
    - At each level, try removing each current feature
    - Remove the feature that gives highest accuracy when removed
    - Stop when accuracy decreases (greedy hill-climbing)
    
    Args:
        total_features: Total number of features available
    """

    current_features = list(range(1, total_features + 1))  # Start with all features
    best_overall_accuracy = 0
    best_overall_features = []
    
    # Evaluate baseline (all features)
    initial_accuracy = leave_one_out_cross_validation(current_features)
    print(f'Using feature(s) {print_feature_set(current_features)} and "random" evaluation, I get an accuracy of {initial_accuracy}%')
    print("\nBeginning search.\n")
    
    best_overall_accuracy = initial_accuracy
    best_overall_features = copy.deepcopy(current_features)

    # Iterate through levels removing one feature at a time
    for level in range(total_features):
        if not current_features:
            break
            
        best_feature_to_remove = None
        best_accuracy_at_level = -1
        
        # Try removing each feature currently in the set
        for feature_id in current_features:
            # Create test set with this feature removed
            test_features = [f for f in current_features if f != feature_id]
            accuracy = leave_one_out_cross_validation(test_features)
            
            print(f"    Using feature(s) {print_feature_set(test_features)} accuracy is {accuracy}%")
            
            # Track best removal at this level
            if accuracy > best_accuracy_at_level:
                best_accuracy_at_level = accuracy
                best_feature_to_remove = feature_id
        
        # Remove the feature that gave best accuracy
        if best_feature_to_remove is not None:
            current_features.remove(best_feature_to_remove)
            print(f"\nFeature set {print_feature_set(current_features)} was best, accuracy is {best_accuracy_at_level}%\n")
            
            # Check if we improved or maintained accuracy
            if best_accuracy_at_level >= best_overall_accuracy:
                best_overall_accuracy = best_accuracy_at_level
                best_overall_features = copy.deepcopy(current_features)
            else:
                # Accuracy decreased so we stop searching
                print("(Warning, Accuracy has decreased!)\n")
                break
        else:
            break
    
    # Report final results
    print(f"Finished search!! The best feature subset is {print_feature_set(best_overall_features)}, which has an accuracy of {best_overall_accuracy}%")

def special_algorithm(total_features):
    """
    Custom search algorithm for extra credit.
    
    PLACEHOLDER: Implement your own algorithm here that is either:
    1. Faster than forward/backward
    2. Gives better results than forward/backward
    
    For Part I submission, this can be left as a stub.
    Must include clear explanation in report for extra credit.
    """

    print("\nRunning Team Blue's Special Algorithm (Beam Search)\n")

    beam_width = 2  #keep only top 2 candidate subsets
    current_beam = [ [] ]  #start with empty subset

    best_overall_subset = []
    best_overall_accuracy = leave_one_out_cross_validation([])

    print(f"Using no features accuracy is {best_overall_accuracy}%")
    print("\nBeginning search.\n")

def main():

    print("Welcome to Team Blue's Feature Selection Algorithm.")
    
    try:
        num_features = int(input("\nPlease enter total number of features: "))
        
        print("\nType the number of the algorithm you want to run.")
        print("  1) Forward Selection")
        
        choice = input("\n")
        
        print()
        
        # Execute selected algorithm
        if choice == '1':
            forward_selection(num_features)
        else:
            print("Invalid selection. Please run the program again and choose 1, 2, or 3.")
    
    except ValueError:
        print("Error: Please enter a valid integer for the number of features.")
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()