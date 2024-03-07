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
    labels = np.hstack((np.ones(nA), 0 * np.ones(nB))) # Class A labeled as 1, Class B as -1
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

def approximation_data(start=-5, end=5.5, step=0.5):
    """
    Generate data for function approximation.

    This function generates x and y values for function approximation tasks. The
    generated data includes a range of x and y values spaced by the specified step.

    Parameters:
    - start (float, optional): Starting value for x and y. Default is -5.
    - end (float, optional): Ending value for x and y. Default is 5.5.
    - step (float, optional): Step size between consecutive x and y values. Default is 0.5.

    Returns:
    - tuple: A tuple containing:
        - x (numpy.ndarray): Array of x values.
        - y (numpy.ndarray): Array of  y values.
    """
    x = np.arange(start, end, step)
    y = np.arange(start, end, step)
    return x, y

def plot_approximation(X, Y, Z):
    """
    Visualize a 3D surface plot for function approximation.

    This function generates a 3D surface plot to visualize the approximation of a function
    based on the provided input data and corresponding output values.

    Parameters:
    - X (numpy.ndarray): 2D array representing the x-coordinate grid.
    - Y (numpy.ndarray): 2D array representing the y-coordinate grid.
    - Z (numpy.ndarray): 2D array representing the corresponding function values.

    Returns:
    - None: The function generates and displays the 3D surface plot.
    """
    # Creating mesh plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.show()



def main():
    classA, classB = generate_data()
    visualize_data(classA, classB)
    shuffled_data, shuffled_labels = shuffle_data(classA, classB)
    print(shuffled_data.shape)
    print(shuffled_labels.shape)

    x, y = approximation_data()
    # Creating meshgrid
    X, Y = np.meshgrid(x, y)
    # Calculating z values
    Z = np.exp(-X**2 * 0.1) * np.exp(-Y.T**2 * 0.1) - 0.5
    plot_approximation(X, Y, Z)



if __name__ == '__main__':
    main()