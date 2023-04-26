#################################################################
# C.Jarne-2023 Analysis Group D. Vidahurre  @cfin               #
# Code is used to plot histograms of EEG from HarMNqEEG dataset # 
# https://doi.org/10.1016/j.neuroimage.2022.119190              #
#################################################################
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to generate the normalized histogram plot and save it as .png

def plot_normalized_histogram(column_data, filename, column_number, subject_id):
    fig, ax = plt.subplots(figsize=(7, 5))
    n_rows      = column_data.shape[0]
    bin_edges   = np.linspace(0.39, 19, n_rows+1)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    
    plt.hist(bin_centers, bins= n_rows,  weights=column_data, density=True, color=plt.cm.viridis(column_number/18))
    ax.set_xlabel('Frequency [Hz]',fontsize = 16)
    ax.set_ylabel('Normalized intensity [Arb. Units]',fontsize = 16)
    ax.set_title('Histogram Ch: '+ str(column_number) + ' Subject: '+str(subject_id))
    plt.xticks(np.arange(0,20.1, 2.5))
    plt.yticks(np.arange(0,0.41, 0.2))
    plt.ylim([0,0.45])
    plt.savefig('plots/column'+str(column_number)+'_.png',dpi=300)
    plt.close()

# Function to read the txt file and call the plot_normalized_histogram function for each column    
def generate_normalized_histograms(filename):
    data = np.loadtxt(filename)
    subject_id = filename.split("_")[-1].split(".")[0]
    for i in range(18):
        column_data = data[:,i]
        plot_normalized_histogram(column_data, filename, i+1, subject_id)
        #print("column_data",column_data)

# Usage example
generate_normalized_histograms('.../histos/F4XW6TBLHF3Q.mat.txt')

