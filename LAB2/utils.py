import numpy as np
import zipfile
import os

def extract_data_to_numpy(file_path):
    """
    Extracts data from a text file and stores it in a NumPy array.

    Parameters:
    - file_path (str): The path to the text file.

    Returns:
    numpy.ndarray: A NumPy array containing the extracted data.
    """
    # Use numpy.loadtxt to read the data from the text file
    data_array = np.loadtxt(file_path)

    return data_array

def extract_zip(zip_file_path, extract_folder):
    """
    Extracts the contents of a zip file into a specified folder.

    Parameters:
    - zip_file_path (str): The path to the zip file to be extracted.
    - extract_folder (str): The path to the folder where the contents of the zip file will be extracted.

    Returns:
    None

    Creates the specified extraction folder if it doesn't exist and extracts all contents of the zip file
    into that folder.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Create the target extraction folder if it doesn't exist
        os.makedirs(extract_folder, exist_ok=True)

        # Extract all contents of the zip file into the specified folder
        zip_ref.extractall(extract_folder)

        print(f"Zip file '{zip_file_path}' successfully extracted to '{extract_folder}'.")


        
def add_noise(data, variance):
    """
    Add Gaussian noise to the input data.

    Parameters:
    - data (numpy.ndarray): Input data.
    - variance (float): Variance of the Gaussian noise.

    Returns:
    numpy.ndarray: Data with added Gaussian noise.
    """
    return data + np.random.normal(0, np.sqrt(variance), len(data))


def threshold_function(x, threshold=0):
    """
    Apply a threshold function to input values.

    Parameters:
    - x (numpy.ndarray): Input values.
    - threshold (float, optional): Threshold value. Values greater than or equal to
      the threshold are set to 1, while values below the threshold are set to -1.
      Default is 0.

    Returns:
    numpy.ndarray: Output values after applying the threshold function.
    """
    return np.where(x >= threshold, 1, -1)


def competitive_learning(data, num_rbf_units, learning_rate_cl, epochs_cl, num_winners=3):
    """Trains a Competitive Learning (CL) network with Radial Basis Function (RBF) units on the input data.

    Parameters:
    - data (numpy.ndarray): Input data for training the CL network. Should be a 2D array.
    - num_rbf_units (int): Number of RBF units in the CL network.
    - learning_rate_cl (float): Learning rate for adjusting RBF node weights during training.
    - epochs_cl (int): Number of training epochs for the CL network.
    - num_winners (int): Number of winning units to be updated during each training step.

    Returns:
    numpy.ndarray:
        The trained RBF nodes representing the learned feature space.

    Raises:
    ValueError: If the input data is not a 2D array.

    The function uses competitive learning to train the RBF network. It initializes random RBF nodes and
    iteratively updates them based on the input data. The goal is to have each RBF node represent a region
    of the input space and adjust its weights to become closer to the training vectors.
    """

     # Check if data is 1D and reshape if necessary
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # Ensure data has the correct shape
    if len(data.shape) != 2:
        raise ValueError("Input data should be a 2D array.")

    num_features = data.shape[1]
    rbf_nodes = np.random.rand(num_rbf_units, num_features)

    for _ in range(epochs_cl):
        np.random.shuffle(data)

        for sample in data:
             # Find the winning units (closest RBF nodes)
            distances = np.linalg.norm(rbf_nodes - sample, axis=1)
            winning_units = np.argsort(distances)[:num_winners]

            # Update the winning units to get closer to the training vector
            for winning_unit in winning_units:
                rbf_nodes[winning_unit] += learning_rate_cl * (sample - rbf_nodes[winning_unit])

    return rbf_nodes


def load_binary_sequence_to_matrix(file, rows=32, cols=84):
    """
    Loads a binary sequence from a file and converts it into a 2D NumPy array.

    This function reads a binary sequence stored in a specified file, where the binary values are separated by commas and potentially spread across multiple lines. It then converts this sequence into a 2D NumPy array of specified dimensions.

    Parameters:
        file (str): The path to the file containing the binary sequence.
        rows (int, optional): The number of rows for the output 2D array. Defaults to 32.
        cols (int, optional): The number of columns for the output 2D array. Defaults to 84.

    Returns:
        np.ndarray: A 2D NumPy array of shape (rows, cols) containing the binary sequence.

    Raises:
        ValueError: If the total number of binary values in the file does not match the product of `rows` and `cols`.
    """
    with open(file, 'r') as file:
        sequence = file.read().replace('\n', '')  # Remove newlines if present
    # Convert the string to a list of integers
    binary_list = [float(bit) for bit in sequence.split(',')]
    
    # Convert the list to a numpy array and reshape into the desired matrix
    binary_matrix = np.array(binary_list).reshape(rows, cols)
    
    return binary_matrix









def main():
    print("utils module")


if __name__ == '__main__':
    main()