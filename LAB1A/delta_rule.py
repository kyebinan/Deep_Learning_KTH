import numpy as np
from utils import loss_function

def delta_learning_online(X, y, epochs = 50, learning_rate = 0.001):
    """
    Train a linear regression model using the Delta learning rule in an online fashion.

    Parameters:
    - X (numpy.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    - y (numpy.ndarray): True labels of shape (m,), where m is the number of samples.
    - epochs (int): Number of training epochs (default: 50).
    - learning_rate (float): Learning rate for adjusting model parameters (default: 0.001).

    Returns:
    - tuple: Lists containing weights and errors at each epoch during training.
      - W (list): List of updated weights at each epoch.
      - errors (list): List of mean squared errors at each epoch.
    """
    weights = np.random.rand(X.shape[1]+1) 
    A = np.ones((X.shape[0],1))
    X = np.concatenate((X, A), axis=1)
    W = []
    errors = []
    for _ in range(epochs):
        for i in range(X.shape[0]):
            weights = weights + learning_rate * (y[i] - np.dot(X[i], weights)) * X[i]
        W.append(weights)
        err = loss_function(X, y, weights)
        errors.append(err)
    return W, errors
