"""
CS170 Project 2: Part 1

Authors: Shreyas, Rishabh, Rehan
"""


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
