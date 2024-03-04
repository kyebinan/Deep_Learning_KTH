import numpy as np 
import matplotlib.pyplot as plt


def generate_data(n=100, mA=np.array([1.0, 0.5]), sigmaA=0.3, mB=np.array([-1.0, 0.0]), sigmaB=0.35):
    """
        Generate synthetic data for two classes (classA and classB).

        Parameters:
        - n (int): Number of data points for each class.
        - mA (numpy.ndarray): Mean of classA in a 2-element array.
        - sigmaA (float): Standard deviation of classA.
        - mB (numpy.ndarray): Mean of classB in a 2-element array.
        - sigmaB (float): Standard deviation of classB.

        Returns:
        - classA (numpy.ndarray): Data points for classA in a 2xN array.
        - classB (numpy.ndarray): Data points for classB in a 2xN array.
    """
    #np.random.seed(42)  # For reproducibility
    classA = np.zeros((2, n))
    classA[0, :] = np.random.randn(1, n) * sigmaA + mA[0]
    classA[1, :] = np.random.randn(1, n) * sigmaA + mA[1]

    classB = np.zeros((2, n))
    classB[0, :] = np.random.randn(1, n) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(1, n) * sigmaB + mB[1]

    return classA, classB


def shuffle_data(classA, classB):
    """
        Shuffle and combine data samples from two classes into a single dataset.

        Parameters:
        - classA (numpy.ndarray): Data points for classA.
        - classB (numpy.ndarray): Data points for classB.

        Returns:
        - shuffled_data (numpy.ndarray): Shuffled and combined data samples.
        - shuffled_labels (numpy.ndarray): Shuffled and combined labels corresponding to the data samples.
    """
    nA = classA.shape[1]
    nB = classB.shape[1]

    data = np.hstack((classA, classB))
    labels = np.hstack((np.ones(nA), -1 * np.ones(nB))) # Class A labeled as 1, Class B as -1
    labels = labels.reshape((1, labels.shape[0]))
    data_set = np.vstack((data, labels))

    np.random.shuffle(data_set)
    n, m = data_set.shape
    shuffled_data, shuffled_labels = data_set[:n-1, :], data_set[n-1, :]
    
    return shuffled_data, shuffled_labels


def visualize_data(classA, classB):
    plt.figure(figsize=(8, 6))
    # Plotting
    plt.scatter(classA[0, :], classA[1, :], color='red', label='Class 1')
    plt.scatter(classB[0, :], classB[1, :], color='blue', label='Class -1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Data for Binary Classification')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    classA, classB = generate_data()
    visualize_data(classA, classB)
    shuffled_data, shuffled_labels = shuffle_data(classA, classB)
    print(shuffled_labels)


if __name__ == '__main__':
    main()