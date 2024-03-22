import numpy as np

class SelfOrganizingMap:
    """
    Self-Organizing Map (SOM) implementation using NumPy.

    Attributes:
        input_size (tuple): Size of the input dimension.
        map_size (tuple): Dimensions of the map (height, width).
        learning_rate (float): Initial learning rate.
        sigma (float): Initial radius of the neighborhood function.
        weights (np.ndarray): The weights of the SOM neurons.
    """

    def __init__(self, input_size, map_size, learning_rate=0.1, sigma=1.0):
        """
        Initialize a new Self-Organizing Map.

        Parameters:
            input_size (int): The dimensionality of the input data.
            map_size (tuple of int): The dimensions (height, width) of the SOM.
            learning_rate (float, optional): Initial learning rate. Defaults to 0.1.
            sigma (float, optional): Initial neighborhood radius. Defaults to 1.0.
        """
        self.input_size = input_size
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = np.random.rand(map_size[0], map_size[1], input_size)

    def find_bmu(self, input_vector):
        """
        Find the Best Matching Unit (BMU) for a given input vector.

        Parameters:
            input_vector (np.ndarray): An input vector to find the BMU for.

        Returns:
            tuple: The (row, column) indices of the BMU.
        """
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_indices = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_indices

    def update_weights(self, input_vector, bmu_indices, iteration, max_iterations):
        """
        Update the weights of the SOM based on the input vector and BMU.

        Parameters:
            input_vector (np.ndarray): The input vector used for updating the weights.
            bmu_indices (tuple): The (row, column) indices of the Best Matching Unit (BMU).
            iteration (int): Current iteration number.
            max_iterations (int): Maximum number of iterations.
        """
        learning_rate = self.learning_rate * (1 - iteration / max_iterations)
        sigma = self.sigma * (1 - iteration / max_iterations)  # Decrease sigma over time

        # Create a grid of indices
        x, y = np.indices((self.map_size[0], self.map_size[1]))
        distance_from_bmu = np.sqrt((x - bmu_indices[0])**2 + (y - bmu_indices[1])**2)

        influence = np.exp(-distance_from_bmu**2 / (2 * sigma**2))
        influence = influence[..., np.newaxis]  # Reshape for broadcasting

        # Update weights
        self.weights += learning_rate * influence * (input_vector - self.weights)

    def train(self, data, max_iterations=100):
        """
        Train the SOM using the provided dataset.

        Parameters:
            data (np.ndarray): The dataset to train the SOM on.
            max_iterations (int, optional): The number of iterations to train for. Defaults to 100.
        """
        for iteration in range(max_iterations):
            for input_vector in data:
                bmu_indices = self.find_bmu(input_vector)
                self.update_weights(input_vector, bmu_indices, iteration, max_iterations)

    def get_weights(self):
        """
        Get the current weights of the SOM.

        Returns:
            np.ndarray: The weights of the SOM.
        """
        return self.weights







def main():
    print("som module")


if __name__ == '__main__':
    main()