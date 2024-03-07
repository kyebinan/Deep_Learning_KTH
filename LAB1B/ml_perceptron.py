import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

TASK1="Classification"
TASK2="Encoder"
TASK3="Regression"
class MultiLayersPerceptron():
    """
    """
    def __init__(self, input_dim, hidden_dim, output_dim, task=TASK1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights_input_to_hidden = np.random.randn(input_dim, hidden_dim)
        self.weights_hidden_to_output = np.random.randn(hidden_dim, output_dim)
        self.bias_hidden = np.random.randn(hidden_dim)
        self.bias_output = np.random.randn(output_dim)
        self.task = task


    def phi(self, x):
        """
        Calculate the hyperbolic tangent activation function.

        The hyperbolic tangent activation function, often denoted as phi(x),
        transforms input values using the formula:
            phi(x) = (2 / (1 + np.exp(-x))) - 1

        Parameters:
        - x (numpy.ndarray or scalar): The input value(s) to be transformed.

        Returns:
        - numpy.ndarray or scalar: The transformed value(s) using the hyperbolic tangent function.

        Note:
        - The hyperbolic tangent function squashes input values to the range [-1, 1].
        - It is commonly used as an activation function in neural networks.
        """
        return (2 / (1 + np.exp(-x))) - 1
    
    def phi_derivate(self, x):
        """
        Calculate the derivative of the hyperbolic tangent activation function.

        The derivative of the hyperbolic tangent activation function, denoted as phi'(x),
        is calculated using the formula:
            phi'(x) = ((1 + phi(x)) * (1 - phi(x))) / 2

        Parameters:
        - x (numpy.ndarray or scalar): The input value(s) for which to calculate the derivative.

        Returns:
        - numpy.ndarray or scalar: The derivative value(s) of the hyperbolic tangent function.
        """
        return ((1 + self.phi(x))*(1 - self.phi(x)))/2

    def loss_function(self, y, y_pred):
        """
        Calculate the mean squared loss between true values and predicted values.

        The mean squared loss measures the average squared difference between
        true values (y) and predicted values (y_pred).

        Parameters:
        - y (numpy.ndarray): True values.
        - y_pred (numpy.ndarray): Predicted values.

        Returns:
        - float: Mean squared loss.
        """
        y = y.reshape((y.shape[0],self.output_dim))
        return ((y - y_pred) ** 2).mean()

    
    def forward_pass(self, X):
        """
        Perform the forward pass through a neural network.

        This function calculates the output of a neural network given input data X.
        The forward pass involves computing the activations of hidden and output layers.

        Parameters:
        - X (numpy.ndarray): Input data with features.

        Returns:
        - tuple: A tuple containing:
            - O (numpy.ndarray): Output layer activations.
            - H (numpy.ndarray): Hidden layer activations.
        """
        H = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        H = self.phi(H)
        O = np.dot(H, self.weights_hidden_to_output) + self.bias_output
        O = self.phi(O)
        return O, H


    def backward_pass(self, y, O, H):
        """
        Perform the backward pass through a neural network for error propagation.

        This function calculates the gradients for weight updates by propagating
        the error from the output layer to the hidden layer through backpropagation.

        Parameters:
        - y (numpy.ndarray): True values.
        - O (numpy.ndarray): Output layer activations.
        - H (numpy.ndarray): Hidden layer activations.

        Returns:
        - tuple: A tuple containing:
            - delta_output (numpy.ndarray): Gradients for weight updates in the output layer.
            - delta_hidden (numpy.ndarray): Gradients for weight updates in the hidden layer.
        """
        # Backpropagation on the output layers
        y = y.reshape((y.shape[0],self.output_dim))
        error = y - O
        delta_output = error * self.phi_derivate(O)
        # Backpropagation on the hidden layers
        error_hidden = delta_output.dot(self.weights_hidden_to_output.T)
        delta_hidden = error_hidden * self.phi_derivate(H)
        return delta_output, delta_hidden


    def weights_update(self, X, H, delta_output, delta_hidden, lr=0.001):
        """
        Update the weights and biases of a neural network based on gradients.

        This function performs weight updates using the gradients calculated during
        the backward pass through backpropagation.

        Parameters:
        - X (numpy.ndarray): Input data with features.
        - H (numpy.ndarray): Hidden layer activations.
        - delta_output (numpy.ndarray): Gradients for weight updates in the output layer.
        - delta_hidden (numpy.ndarray): Gradients for weight updates in the hidden layer.
        - lr (float, optional): Learning rate for controlling the size of weight updates. Default is 0.001.

        Returns:
        - None: The function updates the weights and biases in-place.
        """
        self.weights_hidden_to_output += H.T.dot(delta_output) * lr
        self.bias_output += np.sum(delta_output, axis=0) * lr
        self.weights_input_to_hidden += X.T.dot(delta_hidden) * lr
        self.bias_hidden += np.sum(delta_hidden, axis=0) * lr


    def train(self, X, y, epochs=100, lr = 0.001, batch_size=10):
        """
        Train a neural network using mini-batch gradient descent.

        This function trains a neural network by performing mini-batch gradient descent
        for a specified number of epochs. It monitors the training loss during each epoch.

        Parameters:
        - X (numpy.ndarray): Input data with features.
        - y (numpy.ndarray): True values.
        - epochs (int, optional): Number of training epochs. Default is 100.
        - lr (float, optional): Learning rate for controlling the size of weight updates. Default is 0.001.
        - batch_size (int, optional): Size of mini-batches for mini-batch gradient descent. Default is 10.

        Returns:
        - None: The function updates the neural network parameters in-place.
        """
        errors = []
        n = X.shape[0]//batch_size
        for _ in range(epochs):
            error = 0
            for k in range(n):
                X_batch = X[k*batch_size : (k+1)*batch_size, :]
                y_batch = y[k*batch_size : (k+1)*batch_size]
                # Forward pass
                O, H = self.forward_pass(X_batch)
                # Backward pass
                delta_output, delta_hidden = self.backward_pass(y_batch, O, H)
                # Updating weights
                self.weights_update(X_batch, H, delta_output, delta_hidden, lr=lr)

            # Monitoring loss
            y_pred, _ = self.forward_pass(X)
            error = self.loss_function(y, y_pred)
            errors.append(error)
        self.plot_loss(errors)
        self.plot_decision_boundary(X, y)


    def autoencode(self, patterns, epochs=100, lr = 0.001):
        """
        Train an autoencoder neural network to reconstruct input patterns.

        This function trains an autoencoder neural network using the specified input patterns
        for a specified number of epochs. The autoencoder is designed to reconstruct the input
        patterns, learning a compressed representation in the hidden layer.

        Parameters:
        - patterns (numpy.ndarray): Input patterns for training the autoencoder.
        - epochs (int, optional): Number of training epochs. Default is 100.
        - lr (float, optional): Learning rate for controlling the size of weight updates. Default is 0.001.

        Returns:
        - None: The function updates the autoencoder parameters in-place.
        """
        for _ in range(epochs): 
            # Forward pass
            O, H = self.forward_pass(patterns)
            # Backward pass
            delta_output, delta_hidden = self.backward_pass(patterns, O, H)
            # Updating weights
            self.weights_update(patterns, H, delta_output, delta_hidden, lr=lr)
        self.plot_patterns(patterns, self.make_prediction(patterns))

        

    def make_prediction(self, X, threshold=0.5):
        """
        Make predictions using a trained neural network.

        This function uses a trained neural network to make predictions on input data.
        The predictions are based on the forward pass through the network, and the output
        is adjusted according to the specified threshold and task type.

        Parameters:
        - X (numpy.ndarray): Input data for making predictions.
        - threshold (float, optional): Decision threshold for binary classification tasks.
                                    Default is 0.5.

        Returns:
        - numpy.ndarray: Predicted values based on the neural network's output.
        """
        y_pred, _ = self.forward_pass(X)
        if self.task == TASK1:
            y_pred = (y_pred >= threshold).astype(int)
        elif self.task == TASK2 :
            y_pred = (y_pred >= threshold).astype(int)
            y_pred[y_pred == 0] = -1
        elif self.task == TASK3:
            pass
        else:
            raise ValueError(f"Task must be {TASK1}, {TASK2} or {TASK3}")
        return y_pred

    def plot_decision_boundary(self, X, y):
        """
        Plot the decision boundary of a binary classification neural network.

        This function visualizes the decision boundary of a binary classification neural network
        using contour plots. It also displays the training examples on the plot.

        Parameters:
        - X (numpy.ndarray): Input data with features.
        - y (numpy.ndarray): True values indicating the class labels.

        Returns:
        - None: The function generates a plot.
        """
        plt.figure(figsize=(8, 6))
        plt.xlim(-2, 2)       # Set x-axis limits from -2 to 2
        plt.ylim(-1.5, 1.5)   # Set y-axis limits from -1.5 to 1.5
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        h = 0.01  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the function value for the whole gid
        Z = self.make_prediction(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap='RdBu', alpha=0.7)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')

    def plot_patterns(self, patterns, predictions):
        """
        Visualize input and output patterns of an autoencoder neural network.

        This function generates two sets of subplots: one for the input patterns and one for
        the corresponding output patterns predicted by the autoencoder. Each subplot displays
        an individual pattern as an image.

        Parameters:
        - patterns (numpy.ndarray): Input patterns used for training or testing the autoencoder.
        - predictions (numpy.ndarray): Output patterns predicted by the autoencoder.

        Returns:
        - None: The function generates and displays the subplots.
        """
        cmap = ListedColormap(['black', 'green'])
        plt.figure(figsize=(8, 6))
        for i, pattern in enumerate(patterns):
            plt.subplot(1, 8, i+1)
            plt.imshow(pattern.reshape((2,4)), cmap=cmap)
            plt.title(f'input{i+1}')
            plt.axis('off')
        plt.show()

        plt.figure(figsize=(8, 6))
        for i, pattern in enumerate(predictions):
            plt.subplot(1, 8, i+1)
            plt.imshow(pattern.reshape((2,4)), cmap=cmap)
            plt.title(f'output{i+1}')
            plt.axis('off')
        plt.show()

    
    def plot_loss(self, errors, tilte='Training Loss'):
        """
        Visualize the training loss over epochs.

        This function generates a line plot of the training loss over epochs, providing
        insights into the convergence or divergence of the training process.

        Parameters:
        - errors (list): List of error values at each epoch during training.
        - title (str, optional): Title for the plot. Default is 'Training Loss'.

        Returns:
        - None: The function generates and displays the line plot.
        """
        plt.figure(figsize=(8, 6))
        plt.plot([k for k in range(1, len(errors)+1)], errors)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(tilte)
        plt.grid(True)
        plt.show()


def main():
    print("ml_perceptron module")


if __name__ == '__main__':
    main()