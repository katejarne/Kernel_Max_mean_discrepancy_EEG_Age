#############################################################
# C. Jarne 2024 Analysis Group of D. Vidahurre  @cfin       #
# This code estimates the MMD matrices from the histograms  #
# generated in the simulations for the different conditions #
# input: dir with histograms                                #
# output: MMD matrices                                      #
#############################################################
import Estimation_of_distance_matrices.mmd_def as mmd_def
import numpy as np
import os

# Dir
main_directory = "synthetic-data-grid"

def extract_features(file_path):
    histogram_data = np.loadtxt(file_path)
    histogram, _ = np.histogram(histogram_data, bins=49, density=True)
    bin_heights = histogram / np.diff(_)
    return bin_heights


distances_by_subdir = []

for directory_name in os.listdir(main_directory):
    directory_path = os.path.join(main_directory, directory_name)
    print(directory_name)

    if os.path.isdir(directory_path):

        X_subdir = []

        # Loop over directories
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(directory_path, file_name)
                feature = extract_features(file_path)
                X_subdir.append(feature)

        X_subdir = np.array(X_subdir)
        print(X_subdir.shape)
        # Calculate the distance matrix between all histograms in sub
        num_samples = X_subdir.shape[0]
        print(num_samples)
        distances_subdir = np.zeros((num_samples, num_samples))
        print(distances_subdir.shape)
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                dist = mmd_def.mmd_rbf([X_subdir[i]], [X_subdir[j]])
                distances_subdir[i][j] = dist
                distances_subdir[j][i] = dist

        print("Distance matrices by subdirectory:")
        print(distances_subdir)
        print("shape", distances_subdir.shape)
        # Save the distance matrix to a text file
        file_name = os.path.join(main_directory, f"{directory_name}_distances.txt")
        print(file_name)
        np.savetxt(file_name, distances_subdir)



