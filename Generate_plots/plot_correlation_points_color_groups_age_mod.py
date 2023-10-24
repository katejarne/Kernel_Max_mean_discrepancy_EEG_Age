###################################################################################
#          C.Jarne- 2023 Analysis Group D. Vidahurre  @cfin                       #
#                                                                                 #
# This Python code is used for data analysis and visualization. It specifies      #
# a directory path where files will be analyzed and retrieves a list of all       #
# text files in that directory. The code iterates through each file, extracting   #
# and processing data. It performs a correlation analysis on the data, calculates #
# mean squared error and mean absolute error, and creates plots to visualize the  #
# results. The code also generates scatter plots, violin plots, and 3D scatter    #
# plots. It reads data from a features file, uses this data to group and analyze  #
# the main dataset, and creates scatter plots with labelled groups.               #
# The code saves the generated plots in a directory.                              #
#   input:                                                                        #
#   dir with predicted vs real age files per channel                              #
#   feature file                                                                  #
#   output:                                                                       #
#   plot_+...+png predicted vs real age segregated by site                        #
#   3d_plot_+...+png 3d plot of predicted vs real age segregated by site          #
#   plot_color_violin+...+.png violin plot with uncertanties                      #
###################################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace
from scipy.stats import linregress
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.cm as cm

# Directory containing files to be analyzed
dir_path = "results-predictions/KMER/results/kernel_rbf/corr/"

file_list = [file for file in os.listdir(dir_path) if file.endswith('.txt')]

# Sort the file list numerically based on the number
file_list.sort(key=lambda x: int(x.split('_')[3][6:]))

all_hist_data = []
ch_names_ = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
             'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']
channels_ = np.arange(len(file_list))
etiquetas_str = [ch_names_[i] for i in channels_]

colores = cm.viridis(np.linspace(0, 1, 14))
# Iterate over all files in the directory
for filename in os.listdir(dir_path):
    if not filename.endswith('.txt'):
        continue  # Skip non-text files
    pepe = os.path.splitext(filename)[0]
    pepe = pepe[-3:]
    print(filename.split('_')[3][6:])
    j = int(filename.split('_')[3][6:])
    # read in data from first table
    table1 = np.loadtxt(os.path.join(dir_path, filename), dtype=float)
    data = table1.T
    x = data[0]
    y = data[1]

    # Calculate linear regression and correlation
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    res = stats.spearmanr(x, y)
    print("Correlation ", res[0])
    corr = res[0]
    corr_ = "%.2f" % corr

    # Calculate mean squared error and mean absolute error
    mse_ = mean_squared_error(x, y, squared=False)
    mea_ = mean_absolute_error(x, y)
    mse1__ = "%.2f" % mse_
    mea1__ = "%.2f" % mea_

    # create a dictionary to store data from second table
    groups = {}
    num_elements = {}
    dir_current = os.getcwd()
    up_dir = os.chdir("..")
    print("Directorio actual:", os.getcwd())
    # read in data from second table (Fetures file)
    with open(os.path.join(os.getcwd() + '/input_files/Feature_file/features_ok.txt'), 'r') as f:
        print(os.path.join(os.getcwd() + '/input_files/Feature_file/features_ok.txt'))
        os.chdir(dir_current)
        for k, line in enumerate(f):
            row = line.strip().split('\t')
            if row[1] not in groups:
                groups[row[1]] = {'x': [], 'y': []}
                num_elements[row[1]] = 0
            if row[2] != 'nan' and k < 1926:  # 1949:
                groups[row[1]]['x'].append(float(table1[k][0]))
                groups[row[1]]['y'].append(float(table1[k][1]))
                num_elements[row[1]] += 1

    # create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'aspect': 0.65})

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    num_groups = len(groups)
    cm_subsection = linspace(0, 1, num_groups)

    colors = [cm.tab20b(x) for x in cm_subsection]

    for i, (group, data) in enumerate(groups.items()):
        mse_ = mean_squared_error(data['x'], data['y'], squared=False)
        mea_ = mean_absolute_error(data['x'], data['y'])
        mse__ = "%.2f" % mse_
        mea__ = "%.2f" % mea_
        # to use labels
        # ax.scatter(data['x'], data['y'], label=group+"\nRMSE: "+str(mse__)+" MEA: "+str(mea__),
        # color = colors[i],alpha=0.43)
        ax.scatter(data['x'], data['y'], label=group, s=70, color=colors[i], alpha=0.5)

    xx_ = np.arange(0, 100, 1)
    yy_ = xx_
    plt.plot(xx_, yy_, 'grey', alpha=0.75, label="r= " + str(corr_) + "\nRMSE: " + str(mse1__) + " MAE: " + str(mea1__))
    # Add line to plot
    ax.set_title('Age prediction with MMD Kernel ' + str(pepe) + " Ch: " + etiquetas_str[j], fontsize=20)
    plt.ylim([-10, 101])
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1)
    plt.xlim([0, 100])

    plt.xlabel('Age', fontsize=20)
    plt.ylabel('Predicted age', fontsize=20)

    plt.yticks(np.arange(0.0, 101, 20), fontsize=15)
    plt.xticks(np.arange(0.0, 101, 5), fontsize=15)

    # Legend
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=18)
    plt.savefig('plots/plot_' + os.path.splitext(filename)[0] + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(12, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=-60)
    ax.grid(False)
    marker = [".", ",", "o", "v", "^", "D", ">", "s", "p", "P", "*",
              "h", "H", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for i, (group, data) in enumerate(groups.items()):
        mse_ = mean_squared_error(data['x'], data['y'], squared=False)
        mea_ = mean_absolute_error(data['x'], data['y'])
        mse__ = "%.2f" % mse_
        mea__ = "%.2f" % mea_
        ax.scatter(np.zeros_like(data['y']) + i * 2, data['x'], data['y'],
                   label=group + ": " + str(len(data['x'])), s=50, color=colors[i], alpha=0.75, marker=marker[i])

    xx_ = np.arange(0, 100, 1)
    yy_ = xx_

    ax.plot(np.zeros_like(xx_) + 12.5, xx_, yy_, 'grey', alpha=0.0,
            label="r= " + str(corr_) + "\nMSE: " + str(mse1__) + " MAE: " + str(mea1__))

    ax.set_title('Age Combined MMD Kernel ' + str(pepe) + " Ch: " + etiquetas_str[j], fontsize=20)
    ax.set_ylim([-10, 101])
    ax.set_xlim([0, 28])

    ax.set_ylabel('Age', fontsize=20)
    ax.set_zlabel('Predicted age', fontsize=20)
    ax.set_xlabel('Batch', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0.0, 101, 20))
    # Legend
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=18)
    plt.savefig('plots/3d_plot_' + os.path.splitext(filename)[0] + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Create a figure and a set of axes
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = []

    for i, (group, data) in enumerate(groups.items()):
        diff = np.array(data['y']) - np.array(data['x'])
        mea_ = mean_absolute_error(np.array(data['x']), np.array(data['y']))
        mea__ = "%.2f" % mea_

        # mean and std
        mean = np.mean(diff)
        std = np.std(diff)
        meann = "%.2f" % mean
        stdd = "%.2f" % std
        # labels.append(group+": "+str(len(diff))+"/ RMSE: "+str(stdd))
        labels.append(group + ": " + str(len(diff)) + "/ MAE: " + str(mea__))
        # Generate the violin plot
        parts = ax.violinplot(diff, positions=[i], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)

        # Labels for the x-axis
        ax.set_xticks(range(len(groups)))
        ax.set_yticks(np.arange(-60, 61, 20))
        ax.set_ylim([-61, 61])
        ax.set_xticklabels(list(groups.keys()), rotation=30, fontsize=17, ha="center")
        ax.set_yticklabels(np.arange(-60, 61, 20), fontsize=18)
        # Labels for the y-axis and title
        ax.set_ylabel('[years]', fontsize=18)
        ax.set_title('Predicted - Real Age (per Batch) Ch: ' + etiquetas_str[j], fontsize=18)

        # Limit the range of the x-axis
        ax.set_xlim([-1, len(groups)])
        # Show legend
        ax.legend(labels, fontsize=14, loc='center left', bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()

    ax.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    # Save the figure to a file
    plt.savefig(f'plots/plot_color_violin' + os.path.splitext(filename)[0] + '.png', dpi=200, pad_inches=0,
                transparent=False, format='png')
    plt.close(fig)
print("All plots .png created in file /plots")
