import numpy as np
import random

class LittleHopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        """
        Train the Hopfield network with given patterns using Hebbian learning.

        Parameters:
        - patterns (list of numpy.ndarray): List of patterns to train the network.
        Each pattern should be a 1D numpy array of size `num_neurons`.

        Returns:
        - None

        Notes:
        - The function updates the network's weights based on the given patterns using Hebbian learning.
        - The patterns should be bipolar (-1, 1) or binary (0, 1) values.
        - The diagonal elements of the weight matrix are set to zero to prevent self-connections.
        - The training is done in-place, and the updated weights are stored in the network.
        """
        for pattern in patterns:
            pattern = np.reshape(pattern, (self.num_neurons, 1))
            self.weights += np.outer(pattern, pattern)

        np.fill_diagonal(self.weights, 0)


    def synchronous_update(self, state):
        """
        Perform synchronous update of the Hopfield network.

        Parameters:
        - state (numpy.ndarray): The current state of the network.

        Returns:
        - numpy.ndarray: The updated state.

        Notes:
        - The synchronous update is done by updating all neurons simultaneously.
        - The state parameter is not modified in-place; a new state is returned.
        - The function returns the updated state after synchronous update.
        """
        new_state = np.sign(np.dot(self.weights, state))
        return new_state
    

    def asynchronous_update(self, state):
        """
        Perform asynchronous update of the Hopfield network.

        Parameters:
        - state (numpy.ndarray): The current state of the network.

        Returns:
        - numpy.ndarray: The updated state.

        Notes:
        - The asynchronous update is done by iterating through each neuron and updating its state.
        - The state parameter is modified in-place during the update.
        - The function returns the updated state after asynchronous update.
        """
        for i in range(self.num_neurons):
            state[i] = np.sign(np.dot(self.weights[i, :], state))

        return state
    

    def recall(self, initial_state, update_type='synchronous', max_iterations=100):
        """
        Recall a pattern from the Hopfield network.

        Parameters:
        - initial_state (numpy.ndarray): The initial state of the network.
        - update_type (str): Type of update ('synchronous' or 'asynchronous').
        - max_iterations (int): Maximum number of iterations for asynchronous update.

        Returns:
        - numpy.ndarray: The recalled state.

        Notes:
        - The recall process is performed based on the trained weights of the Hopfield network.
        - The update_type parameter determines whether to use 'synchronous' or 'asynchronous' update.
        - For asynchronous update, the max_iterations parameter controls the maximum number of iterations.
        - The function returns the recalled state after the recall process.        
        """
        recalled_state = initial_state.copy()

        if update_type == 'synchronous':
            for _ in range(max_iterations):
                new_state = self.synchronous_update(recalled_state)
                if np.array_equal(new_state, recalled_state):
                    break
                recalled_state = new_state

        elif update_type == 'asynchronous':
            for _ in range(max_iterations):
                for i in range(self.num_neurons):
                    new_state_i = self.asynchronous_update(recalled_state.copy())[i]
                    recalled_state[i] = new_state_i

                if np.array_equal(new_state_i, recalled_state[i]):
                    break

        return recalled_state
    

if __name__ == "__main__":
    print("================================================")
    print("|This is the utils module for the notebook 3.  |")
    print("================================================")