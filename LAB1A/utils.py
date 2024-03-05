import numpy as np
import matplotlib.pyplot as plt

def loss_function(X, y, weights, bias=0):
    """
    Calculate the mean squared loss for a linear regression model.

    Parameters:
    - X (numpy.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    - y (numpy.ndarray): True labels of shape (m,), where m is the number of samples.
    - weights (numpy.ndarray): Coefficients for each feature in the linear model.

    Returns:
    - float: Mean squared loss between predicted and true labels.
    """
    tmp = np.dot(X, weights) + bias
    y_hat = np.where(tmp >= 0, 1, -1)
    errors = y - y_hat
    return np.mean(errors ** 2)


def make_prediction(weights, bias, features):
    """
    Make a binary prediction using a linear classifier.

    Parameters:
    - weights (numpy.ndarray): Coefficients for each feature in the linear model.
    - bias (float): Bias term in the linear model.
    - features (numpy.ndarray): Input features for making predictions.

    Returns:
    - int: Binary prediction (1 or -1) based on the linear model.
    """
    y = np.dot(features, weights) + bias
    return 1 if y >= 0 else -1 


def visualize_loss_function(errors, title):
    """
    Visualize the error over epochs for a training process.

    Parameters:
    - errors (list): List of error values at each epoch.
    - title (str): Title for the plot.

    Returns:
    - None: Displays a line plot of the error over epochs.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([k for k in range(1, len(errors)+1)], errors)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    print("utils module")


if __name__ == '__main__':
    main()