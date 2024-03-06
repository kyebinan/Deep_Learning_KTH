import numpy as np
import matplotlib.pyplot as plt 


def generate_data(ndata=300, mA=np.array([1.0, 0.3]), sigmaA=0.2, mB=np.array([0.0, -0.1]), sigmaB=0.3):
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
    # Generate data for class A
    classA_x = np.concatenate([
        np.random.randn(round(0.5 * ndata)) * sigmaA - mA[0],
        np.random.randn(round(0.5 * ndata)) * sigmaA + mA[0]
    ])
    classA_y = np.random.randn(ndata) * sigmaA + mA[1]
    classA = np.vstack((classA_x, classA_y)).T  # Transpose to get [ndata, 2] shape

    # Generate data for class B
    classB_x = np.random.randn(ndata) * sigmaB + mB[0]
    classB_y = np.random.randn(ndata) * sigmaB + mB[1]
    classB = np.vstack((classB_x, classB_y)).T  # Transpose to get [ndata, 2] shape

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
    nA = classA.shape[0]
    nB = classB.shape[0]

    data = np.vstack((classA, classB))
    labels = np.hstack((np.ones(nA), -1 * np.ones(nB))) # Class A labeled as 1, Class B as -1
    indices = np.random.permutation(data.shape[0])
    shuffled_data = data[indices,:]
    shuffled_labels = labels[indices]
    
    return shuffled_data, shuffled_labels


def visualize_data(classA, classB):
    """
    Visualize 2D data points for binary classification.

    Parameters:
    - classA (numpy.ndarray): Data points for class 1 in a 2xN array.
    - classB (numpy.ndarray): Data points for class -1 in a 2xN array.

    Returns:
    - None: Displays a scatter plot of the data points with different colors for each class.
    """
    plt.figure(figsize=(8, 6))
    # Plotting
    plt.scatter(classA[:, 0], classA[:, 1], color='red', label='Class 1')
    plt.scatter(classB[:, 0], classB[:, 1], color='blue', label='Class -1')
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
    print(shuffled_data.shape)
    print(shuffled_labels.shape)


if __name__ == '__main__':
    main()