#############################################################################
#              C. Jarne 204 Analysis Group of D. Vidahurre  @cfin          #
#                                                                           #
# Code to generate  scatter plots of the actual ages vs. predicted age      #
# where the points will be colored according to the type of kernel used.    #
# In addition, a linear fit line is added and the Pearson correlation       #
# value is displayed in the figure caption.                                 #
# The figure is saved in .png format.                                       #
# input: dir with prediction files ( be aware of file names)                #
# output:                                                                   #
# plot_+..+png        predicted vs. real age                                #
# plot_hist_+..+.png  histogram with errors                                 #
# delta_+..+.png      delta: predicted -real age vs. ral age                #
#############################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy import stats
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score
import matplotlib.cm as cm
# Channel order: ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
#                  'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']

# path with files with predicted vs real age

dir_path = "Results_files/KMER_gamma_free_0.1-1-10/"

colores = ["#5ec962", "#5ec962"]  # , "#3b528b", "#440154"]
#colores = cm.viridis(np.linspace(0, 1, 3))

for filename in os.listdir(dir_path):
    if filename.startswith("KMER_prediction_") and filename.endswith(".txt"):
    #if filename.startswith("KRR_prediction_") and filename.endswith(".txt"):
        # Read input file
        pepe = filename.split(".")
        kernel_label = pepe[0]
        kernel_label_2 = kernel_label[0:10]
        kernel_label ="rbf"# kernel_label[-3:]
        
        data_ = np.genfromtxt(dir_path + filename, delimiter='\t')
        data = data_.T
        x = data[0] # np.exp(data[0])
        y = data[1] # np.exp(data[1])
        
        # Remove NaN values from data
        not_nan_indices = np.logical_and(~np.isnan(x), ~np.isnan(y))
        x = x[not_nan_indices]
        y = y[not_nan_indices]
        
        # Select pairs with x value less than 100
        selected_indices = x > np.log(5)
        x_selected = x[selected_indices]
        y_selected = y[selected_indices]

        score = r2_score(x_selected, y_selected)
        # Calculate linear regression and correlation
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        res = stats.spearmanr(x_selected, y_selected)
        print("Correlation ", res[0])
        corr = res[0]
        corr_ = "%.2f" % score  # change according to what you want to show in plot
        
        # Calculate mean squared error and mean absolute error
        mse_ = mean_squared_error(x, y, squared=False)
        mea_ = mean_absolute_error(x, y)
        mse__ = "%.2f" % mse_
        mea__ = "%.2f" % mea_

        # Plot predicted age vs real age
        fig, ax = plt.subplots(figsize=(10,6))
        plt.axhline(0, color='grey', linestyle='dashed', linewidth=1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set color depending on kernel type
        if str(kernel_label) == "rbf":
            plt.scatter(x, y, color=colores[1], alpha=0.5,
                        label=r'Channel T3'+ "\n$R^2$= "+str(corr_))
        if str(kernel_label) == "lin":
            plt.scatter(x, y, color="#5ec962", alpha=0.5,
                        label=r'Channel T3'+ "\n$R^2$= "+str(corr_))
        if str(kernel_label) == "pol":
            plt.scatter(x, y, color="#5ec962", alpha=0.5,
                        label=r'Channel T3' + "\n$R^2$= "+str(corr_))
        
        xx = np.arange(0, 98, 1)
        #yy = slope * xx + intercept
        yy = xx
        plt.plot(xx, yy, color="gray",linestyle='dashed')
        # Add linear fit to plot
        #plt.xlabel('Log(Age)', fontsize=18)
        plt.xlabel('Age', fontsize=18)
        #plt.ylabel('KMER predicted Log(age)', fontsize=18)
        plt.ylabel('KMER predicted age', fontsize=18)
        plt.legend(loc='upper left', fontsize=18)
        plt.ylim([-10, 100])
        plt.xlim([0, 102])
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        #plt.xticks(np.arange(0, 101, 10), fontsize=18)
        figname = "plots/plot_"+str(kernel_label_2)+"_corr_"+filename.split(".")[0]+"_.png"
        plt.savefig(figname, dpi=200)
        plt.close()

        # Saving results in files

        with open("plots/correlations_"+str( kernel_label)+".txt", "a") as f:
            f.write(filename + "\t" + str(corr) + "\n")
        with open("plots/error_mse_"+str( kernel_label)+".txt", "a") as f:
            f.write(filename + "\t" + str(mse_) + "\n")
        with open("plots/error_mae_"+str( kernel_label)+".txt", "a") as f:
            f.write(filename + "\t" + str(mea_) + "\n")
        with open("plots/r2score_"+str(kernel_label)+".txt", "a") as f:
            f.write(filename + "\t" + str(score)+ "\n")

        differences = y_selected-x_selected
        
        # Calculate mean squared error and mean absolute error
        mean = np.mean(differences)
        std = np.std(differences)
        
        # Histogram with the differences
        
        fig = plt.figure(figsize=(10, 8))

        if str(kernel_label) == "lin":
            plt.hist(differences, bins=100, alpha=0.5, color="#3b528b", label="Delta")
        if str(kernel_label) == "pol":
            plt.hist(differences, bins=100, alpha=0.5, color="#440154", label="Delta")
        if str(kernel_label) == "rbf":
            plt.hist(differences, bins=100, alpha=0.5, color="#5ec962", label="Delta")

        plt.xlabel('Predicted- Real Age', fontsize=18)
        plt.ylabel('Subjects', fontsize=18)
        plt.xlim([-100, 100])

        plt.axhline(0, color='grey', linestyle='dashed', linewidth=1)
        plt.legend(loc='upper left', fontsize=20)

        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)

        # Add mean and STD to the plot
        plt.savefig("plots/plot_hist_"+filename.split(".")[0]+'.png', dpi=300, pad_inches=0, format='png')
        plt.close()

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_selected, differences)
        res2 = stats.spearmanr(x_selected, differences)
        corr2 = res2[0]
        corr2_ = "%.2f" % corr2
        
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #   spine.set_visible(False)
        if str(kernel_label) == "rbf":
            plt.scatter(x_selected, differences, color="#5ec962", alpha=0.5,
                        label="T3 MMD Kernel "+str(kernel_label)+"\n Delta Correlation = "+str(corr2_))
        if str(kernel_label) == "lin":
            plt.scatter(x_selected, differences, color="#3b528b", alpha=0.5,
                        label="MMD Kernel "+str(kernel_label)+"\n Delta Correlation = "+str(corr2_))
        if str(kernel_label) == "pol":
            plt.scatter(x_selected, differences, color="#440154", alpha=0.5,
                        label="MMD Kernel "+str(kernel_label)+"\n Delta Correlation = "+str(corr2_))
      
        # Add linear fit to plot
        plt.ylim([-100, 100])
        plt.xticks(np.arange(0, 101, 10), fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Age", fontsize=20)
        plt.ylabel("(Predicted - Real) age", fontsize=20)
        plt.axhline(0, color='grey', linestyle='dashed', linewidth=1)
        plt.legend(loc='upper right', fontsize=18)
        figname = "plots/plot_"+filename.split(".")[0]+"_error_.png" 
        fig.tight_layout()
        plt.savefig(figname, dpi=200)

        with open("plots/delta_"+str(kernel_label)+".txt", "a") as f:
            f.write(filename + "\t" + str(corr2) + "\n")
        
print("All plots created in file /plots .png")        
        
