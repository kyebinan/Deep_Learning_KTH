import numpy as np
import matplotlib.pyplot as plt 

TASK1="Classification"
TASK2="Regression"
class MultiLayersPerceptron():
    """
    """
    def __init__(self, input_dim, hidden_dim, task=TASK1):
        self.weights_input_to_hidden = np.random.randn(input_dim, hidden_dim)
        self.weights_hidden_to_output = np.random.randn(hidden_dim, 1)
        self.bias_hidden = np.random.randn(hidden_dim)
        self.bias_output = np.random.randn(1)
        self.task = task


    def phi(self, x):
        return (2 / (1 + np.exp(-x))) - 1
    
    def phi_derivate(self, x):
        return ((1 + self.phi(x))*(1 - self.phi(x)))/2

    def loss_function(self, y, y_pred):
        return ((y - y_pred) ** 2).mean()

    
    def forward_pass(self, X):
        H = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        H = self.phi(H)
        O = np.dot(H, self.weights_hidden_to_output) + self.bias_output
        O = self.phi(O)
        return O, H


    def backward_pass(self, y, O, H):
        # Backpropagation on the output layers
        y = y.reshape((y.shape[0],1))
        error = y - O
        delta_output = error * self.phi_derivate(O)
        # Backpropagation on the hidden layers
        error_hidden = delta_output.dot(self.weights_hidden_to_output.T)
        delta_hidden = error_hidden * self.phi_derivate(H)
        return delta_output, delta_hidden


    def wieghts_update(self, X, H, delta_output, delta_hidden, lr=0.001):
        self.weights_hidden_to_output += H.T.dot(delta_output) * lr
        self.bias_hidden += np.sum(delta_output, axis=0) * lr
        self.weights_input_to_hidden += X.T.dot(delta_hidden) * lr
        self.bias_output += np.sum(delta_output, axis=0) * lr


    def train(self, X, y, epochs=100, lr = 0.001, batch_size=10):
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
                self.wieghts_update(X_batch, H, delta_output, delta_hidden, lr=lr)

            # Monitoring loss
            y_pred, _ = self.forward_pass(X)
            error = self.loss_function(y, y_pred)
            errors.append(error)
        #self.plot_loss(errors)
        self.plot_decision_boundary(X, y)
        

    def make_prediction(self, X, threshold=0.5):
        y_pred, _ = self.forward_pass(X)
        if self.task == TASK1:
            y_pred = (y_pred >= threshold).astype(int)
        elif self.task == TASK2:
            pass
        else:
            raise ValueError(f"Task must be {TASK1} or {TASK2}")
        return y_pred

    def plot_decision_boundary(self, X, y):
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
    
    def plot_loss(self, errors, tilte='Training Loss'):
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