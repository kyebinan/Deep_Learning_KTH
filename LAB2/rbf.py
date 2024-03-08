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
        print(activations.shape)
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
        i = np.linalg.inv(phiX.T @ phiX) @ phiX.T
        self.weights = np.linalg.inv(phiX.T @ phiX) @ phiX.T @ y
        

    def make_prediction(self, X):
        pass






def main():
    print("radial basis function network module")


if __name__ == '__main__':
    main()