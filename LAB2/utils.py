import numpy as np 

def add_noise(data, variance):
    """
    Add Gaussian noise to the input data.

    Parameters:
    - data (numpy.ndarray): Input data.
    - variance (float): Variance of the Gaussian noise.

    Returns:
    numpy.ndarray: Data with added Gaussian noise.
    """
    return data + np.random.normal(0, np.sqrt(variance), len(data))


def threshold_function(x, threshold=0):
    """
    Apply a threshold function to input values.

    Parameters:
    - x (numpy.ndarray): Input values.
    - threshold (float, optional): Threshold value. Values greater than or equal to
      the threshold are set to 1, while values below the threshold are set to -1.
      Default is 0.

    Returns:
    numpy.ndarray: Output values after applying the threshold function.
    """
    return np.where(x >= threshold, 1, -1)


def main():
    print("utils module")


if __name__ == '__main__':
    main()