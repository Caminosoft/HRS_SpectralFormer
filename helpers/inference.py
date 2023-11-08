import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import os 
from ..demo import *


def perform_inference(model, label_test_loader, label_true_loader, height, width, total_pos_true, color_matrix, label, model_state_path, dataset_name):
    # Load the trained model
    model.load_state_dict(torch.load(model_state_path))
    model.eval()

    # Validate and get performance metrics
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    # Test and create prediction matrix
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1

    plt.subplot(1, 1, 1)
    plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    plt.xticks([])
    plt.yticks([])
    plt.show()

    output_dir = './output/'  # Directory to store output files
    os.makedirs(output_dir, exist_ok=True)  # Creates the output directory if it doesn't exists
    mat_filename = output_dir + 'matrix_' + dataset_name + '.mat'
    png_filename = output_dir + 'classification_' + dataset_name + '.png'

    savemat(mat_filename, {'P': prediction_matrix, 'label': label})
    plt.savefig(png_filename)
    print("\n-------------Successfully performed inference---------------------")
    return OA2, AA_mean2, Kappa2, AA2