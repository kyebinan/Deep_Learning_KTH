import numpy as np

class SelfOrganizingMap:
    def __init__(self, input_size, map_size, learning_rate=0.1, sigma=1.0):
        self.input_size = input_size
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.sigma = sigma

        # Initialize weights with random values
        self.weights = np.random.rand(map_size[0], map_size[1], input_size)

    def find_bmu(self, input_vector):
        # Calculate the distances between the input vector and all neurons
        distances = np.linalg.norm(self.weights - input_vector, axis=2)

        # Find the Best Matching Unit (BMU)
        bmu_indices = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_indices

    def update_weights(self, input_vector, bmu_indices, iteration, max_iterations):
        # Update weights based on the input vector and BMU
        learning_rate = self.learning_rate * (1 - iteration / max_iterations)
        influence = np.exp(-np.linalg.norm(np.array(bmu_indices) - self.map_size / 2) / (2 * self.sigma ** 2))

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                weight_update = learning_rate * influence * (input_vector - self.weights[i, j])
                self.weights[i, j] += weight_update

    def train(self, data, max_iterations=100):
        for iteration in range(max_iterations):
            for input_vector in data:
                bmu_indices = self.find_bmu(input_vector)
                self.update_weights(input_vector, bmu_indices, iteration, max_iterations)

    def get_weights(self):
        return self.weights








def main():
    print("som module")


if __name__ == '__main__':
    main()