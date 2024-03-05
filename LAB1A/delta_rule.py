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



def delta_learning_batch_step(weights, X_batch, y_batch, learning_rate = 0.001):
    """
    Update the weights of a linear regression model using the Delta learning rule in batch mode.

    Parameters:
    - weights (numpy.ndarray): Current weights of the linear model.
    - X_batch (numpy.ndarray): Input features for the batch update.
    - y_batch (numpy.ndarray): True labels corresponding to the batch.
    - learning_rate (float): Learning rate for adjusting model parameters (default: 0.001).

    Returns:
    - numpy.ndarray: Updated weights after the batch update.
    """
    output = np.dot(X_batch, weights)
    errors = y_batch - output
    weights += learning_rate * np.dot(X_batch.T, errors)
    return weights


def delta_learning_batch(X, y, epochs = 50, learning_rate = 0.001, batch_size=10):
    """
    Train a linear regression model using the Delta learning rule in batch mode.

    Parameters:
    - X (numpy.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    - y (numpy.ndarray): True labels of shape (m,), where m is the number of samples.
    - epochs (int): Number of training epochs (default: 50).
    - learning_rate (float): Learning rate for adjusting model parameters (default: 0.001).
    - batch_size (int): Size of the batches for batch updates (default: 10).

    Returns:
    - tuple: Lists containing weights and errors at each epoch during training.
      - W (list): List of updated weights at each epoch.
      - errors (list): List of mean squared errors at each epoch.
    """
    weights = np.random.rand(X.shape[1]+1)
    A = np.ones((X.shape[0],1))
    X = np.concatenate((X, A), axis=1)
    W = []
    n = X.shape[0]//batch_size
    errors = []
    for _ in range(epochs):
        for k in range(n):
            X_batch = X[k*batch_size : (k+1)*batch_size, :]
            y_batch = y[k*batch_size : (k+1)*batch_size]
            weights = delta_learning_batch_step(weights, X_batch, y_batch, learning_rate=learning_rate)
            W.append(weights)
        err = loss_function(X,y, weights)
        errors.append(err)
    return W, errors


def main():
    print("delta rule module")


if __name__ == '__main__':
    main()