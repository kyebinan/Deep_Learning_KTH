import numpy as np 


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


def perceptron_learning_step(weights, bias, features, label, learning_rate = 0.001):
    """
    Update the weights and bias of a perceptron in a single learning step.

    Parameters:
    - weights (numpy.ndarray): Current weights of the perceptron.
    - bias (float): Current bias of the perceptron.
    - features (numpy.ndarray): Input features for the learning step.
    - label (int): True label for the given input features (-1 or 1).
    - learning_rate (float): Learning rate for adjusting the perceptron parameters.

    Returns:
    - tuple: Updated weights and bias after the learning step.
    """
    pred = make_prediction(weights, bias, features)
    if pred == label:
        return weights, bias
    else :
        for i in range(len(weights)):
            weights[i] += learning_rate * features[i] * (label - pred)
        bias += learning_rate * (label - pred)
        return weights, bias
    

def perceptron_learning(X, y, epochs = 20, learning_rate = 0.001):
    """
    Train a perceptron model using the perceptron learning algorithm.

    Parameters:
    - X (numpy.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    - y (numpy.ndarray): True labels of shape (m,), where m is the number of samples.
    - epochs (int): Number of training epochs (default: 20).
    - learning_rate (float): Learning rate for adjusting perceptron parameters (default: 0.001).

    Returns:
    - tuple: Lists containing weights and biases at each step during training.
      - W (list): List of updated weights at each learning step.
      - B (list): List of updated biases at each learning step.
    """
    weights = np.random.rand(X.shape[1])
    bias = 0.0
    W = []
    B = []
    for _ in range(epochs):
        for i in range(X.shape[0]):
            weights, bias = perceptron_learning_step(weights, bias, X[i], y[i], learning_rate)
            W.append(weights)
            B.append(bias)

    return W, B


def main():
    print("Perceptron learning rule")


if __name__ == '__main__':
    main()