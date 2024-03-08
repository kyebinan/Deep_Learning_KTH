import numpy as np
import matplotlib.pyplot as plt 

class RBFNetwork():
    """
    """
    def __init__(self, centers, variance):
        self.centers = centers
        self.variance = variance
        self.weights = np.random.randn(centers.shape[0],centers.shape[1])


    def phi(self, X):
        """
        Compute RBF activations for each input.

        Parameters:
        - X: Input data matrix (numpy array, shape: [n_samples, n_features])
        - centers: RBF centers (numpy array, shape: [n_centers, n_features])
        - variance: RBF variance (scalar)

        Returns:
        - activations: RBF activations for each input (numpy array, shape: [n_samples, n_centers])
        """
        # Compute squared Euclidean distances between each input and each center
        distances = np.sum((X[:, np.newaxis, :] - self.centers) ** 2, axis=2)

        # Compute RBF activations using the Gaussian RBF formula
        activations = np.exp(-distances / (2 * self.variance**2))
        return activations
    
    def batch_least_squares(self, X, y):
        """
        Solve linear regression using the normal equations.

        Parameters:
        - X: Input feature matrix (numpy array, shape: [n_samples, n_features])
        - y: Target values (numpy array, shape: [n_samples])
        """
        # Compute the optimal parameters using the normal equations
        phiX = self.phi(X)
        self.weights = np.linalg.inv(phiX.T @ phiX) @ phiX.T @ y
        

    def make_prediction(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Input data matrix (numpy array, shape: [n_samples, n_features])

        Returns:
        - predictions: Predicted outputs (numpy array, shape: [n_samples])
        """
        phiX = self.phi(X)
        predictions = phiX @ self.weights
        return predictions
    
    def plot_prediction(self, X, y, y_pred, title):
        """
        Plot the true function, predicted function, and weights for the given approximation.

        Parameters:
        - X: Input data matrix (numpy array, shape: [n_samples, n_features])
        - y: True function values (numpy array, shape: [n_samples])
        - y_pred: Predicted function values (numpy array, shape: [n_samples])
        - title: Title for the plot (string)

        Returns:
        - None

        Description:
        This function generates a visual comparison between the true function, predicted function, 
        and weights for a given approximation. It plots two subplots: 
        1. The left subplot displays the true function and the predicted function.
        2. The right subplot displays the weights associated with the approximation centers.
        """
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 2, 1)
        plt.plot(X, y, label='True Function')
        plt.plot(X, y_pred, label='Predicted Function',)
        plt.title(f'{title} Approximation')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.centers, self.weights, marker='o')
        plt.title(f'Weights for {title} Approximation')

        plt.show()






def main():
    print("radial basis function network module")


if __name__ == '__main__':
    main()