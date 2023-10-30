import sys

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def load_mat_file(file_path):
    """
    Load a .mat file and return its content.

    Parameters:
    - file_path (str): The path to the .mat file.

    Returns:
    - dict: Dictionary containing the variables in the .mat file.
    """
    data = loadmat(file_path)
    return data


def compare_mat_files(file1_path, file2_path):
    """
    Compare two .mat files.

    Parameters:
    - file1_path (str): The path to the first .mat file.
    - file2_path (str): The path to the second .mat file.

    Returns:
    - bool: True if the contents of the files are the same, False otherwise.
    """
    data1 = load_mat_file(file1_path)
    data2 = load_mat_file(file2_path)

    # Compare variables or other aspects of the files
    # Modify this part based on your specific requirements
    return data1 == data2


# Example usage
file1_path = "/home/einn10184/PycharmProjects/IEEE_TGRS_SpectralFormer/data/IndianPine.mat"
file2_path = "/home/einn10184/PycharmProjects/IEEE_TGRS_SpectralFormer/data/combined_data.mat"

# Load the content of the .mat files
data1 = load_mat_file(file1_path)
data2 = load_mat_file(file2_path)


# Compare the content of the .mat files
# are_files_equal = compare_mat_files(file1_path, file2_path)

def explain(input):
    print("input has TR ", input['TR'].shape)
    print("input has TE ", input['TE'].shape)
    print("input has input ", input['input'].shape)
    np.set_printoptions(threshold=sys.maxsize)
    # print(f"test data is {input['TR']}")
    data = input['TE']
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='rainbow')
    plt.title("tr data")
    plt.axis('on')
    plt.colorbar()
    plt.show()


# Print results
print("Content of file1.mat:", explain(data1))
# print("Content of file2.mat:", explain(data2))
# print("Are the files equal?", are_files_equal)
