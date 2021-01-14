"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    
    # From slides (and exericse)
    # Dice = 2 * intersection(x, y) / sum(x, y)
    # Where 
    # Intersection = np.sum(a*b)
    # Note additional logic for scenario where denominator == 0
    
    # Convert to binary: here, we're not going to make a distinction between classes beyond "0" and "not 0"
    a, b = np.where(a > 0, 1, 0), np.where(b > 0, 1, 0)
    
    if (np.sum(a) + np.sum(b)) == 0:
        return -1

    return (np.sum(a * b) * 2.0) / (np.sum(a) + np.sum(b))
    

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    # <YOUR CODE GOES HERE>
    
    # From slides (and exericse)
    # Jaccard = intersection(x, y) / union(x, y)
    # Where
    # Union = sum - intersection
    # TODO remove Similar logic for denominator == 0  
    a, b = np.where(a > 0, 1, 0), np.where(b > 0, 1, 0)
    
    if ((np.sum(a) + np.sum(b)) - np.sum(a * b)) == 0:
        return -1
    
    return  np.sum(a * b) / ((np.sum(a) + np.sum(b)) - np.sum(a * b))