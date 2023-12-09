import numpy as np

def normilize(arr):
    """
    Changes the mean and standard deviation of an arry

    Args:
        arr (np.array): orginal array
        mean (_type_): new mean of array
        std (_type_): new standard deviation of array

    Returns:
        np.array: new array with updated mean and standard deviation
    """
    new_arr = np.copy(arr)
    
    # Update mean
    new_arr = (new_arr - np.mean(arr)) / np.std(arr)
    
    return new_arr