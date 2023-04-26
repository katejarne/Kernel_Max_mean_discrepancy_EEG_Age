
###############################################################################
#              C. Jarne 2023 Analysis Group of D. Vidahurre  @cfin            #
# Code to generate  scatterplots of the actual ages vs. the predicted ages    #  
# where the points will be colored according to the type of kernel used.      # 
# In addition, a linear fit line is added and the Pearson correlation         #      
# value is displayed in the figure caption.                                   # 
# The figure is saved in .png format.                                         #
###############################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy import stats
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score

#path with files with predicted vs real age (combined or individual)
#dir_path="..."
#dir_path="/home/kathy/Desktop/kernel-methods/my_code/code_kernel_mmd/clean_code_git_EEG/"
# Loop through all .txt files in the directory

for filename in os.listdir(dir_path):

    if filename.endswith(".txt"):
    
        # Read input file
        
        pepe= filename.split(".")
        kernel_label   =pepe[0]
        kernel_label_2 = kernel_label[0:10]
        kernel_label   =kernel_label[-3:]
        
        data_= np.genfromtxt(dir_path + filename, delimiter='\t')
        data=data_.T
        x = data[0]
        y = data[1]
        
        # Remove NaN values from data
        not_nan_indices = np.logical_and(~np.isnan(x), ~np.isnan(y))
        x = x[not_nan_indices]
        y = y[not_nan_indices]
        
        # Select pairs with x value less than 100
        selected_indices = x <95
        x_selected = x[selected_indices]
        y_selected = y[selected_indices]
        score =r2_score(x_selected,y_selected)
        # Calculate linear regression and correlation
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        res = stats.spearmanr(x_selected,y_selected) #correlation spearman
        print("Correlation ",res[0])
        corr  =res[0]
        corr_ ="%.2f" % corr
        
        # Calculate mean squared error and mean absolute error
        mse_ =mean_squared_error(x, y, squared=False)
        mea_= mean_absolute_error(x, y)
        mse__="%.2f" % mse_
        mea__="%.2f" % mea_
        
        # Plot predicted age vs real age
         
        fig=plt.figure(figsize=(10,6))
        plt.axhline(0 , color='grey', linestyle='dashed', linewidth=1)
        
        #plt.scatter(x, y, color="grey",alpha = 0.5,label="Kernel "+str(kernel_label)+"\n All regions combined R spearmanr: "+str(corr_))
        
          # Set color depending on kernel type
        if str(kernel_label)=="rbf":
            plt.scatter(x, y, color="#5ec962",alpha = 0.5,label="Combined MMD Kernel "+str(kernel_label)+"\n r= "+str(corr_)+"\nRMSE: "+str(mse__)+" MAE: "+str(mea__))#rbf
        if str(kernel_label)=="cos":
            plt.scatter(x, y, color="#fde725",alpha = 0.5,label="Combined MMD Kernel "+str(kernel_label)+"\n r= "+str(corr_)+"\nRMSE: "+str(mse__)+" MAE: "+str(mea__))#cos
        if str(kernel_label)=="lin":
            plt.scatter(x, y, color="#3b528b",alpha = 0.5,label="Combined MMD Kernel "+str(kernel_label)+"\nr= "+str(corr_)+"\nRMSE: "+str(mse__)+" MAE: "+str(mea__))#lin
        if str(kernel_label)=="pol":
            plt.scatter(x, y, color= "#440154",alpha = 0.5,label="Combined MMD Kernel "+str(kernel_label)+"\n r= "+str(corr_)+"\nRMSE: "+str(mse__)+" MAE: "+str(mea__))#pol
        
        xx = np.arange(0, 100, 1)
        yy = xx
        plt.plot(xx, yy,'grey',alpha=0.75)
        # Add linear fit to plot   
        #plt.plot(x, intercept + slope*x, 'grey',alpha=0.75, label='Linear Fit')
        plt.xlabel('Age',fontsize = 18)
        plt.ylabel('Predicted age',fontsize = 18)
        plt.legend(loc='upper left',fontsize = 18)
        #plt.title("Kernel "+str(kernel_label)+' R: {:.3f}'.format(r_value)+" (Pearson)")
        plt.ylim([-10,100])
        plt.xlim([0,102])
        plt.xticks(np.arange(0,100, 5))
        figname = "plots/plot_"+str(kernel_label_2)+"_corr_"+filename.split(".")[0]+"_.png" 
        plt.savefig(figname,dpi=200)
        plt.close()

        #Saving results in files for bar plots later

        with open("plots/correlations_"+str( kernel_label)+".txt", "a") as f:
            f.write(filename + "\t" + str(corr) + "\n")
        with open("plots/error_mse_"+str( kernel_label)+".txt", "a") as f:
            f.write(filename + "\t" + str(mse_) + "\n")
        with open("plots/error_mae_"+str( kernel_label)+".txt", "a") as f:
            f.write(filename + "\t" + str(mea_) + "\n")
        with open("plots/r2score_"+str(kernel_label)+".txt", "a") as f:
            f.write(filename + "\t" + str(score)+ "\n")
        """
        #use if combined MMD directory
        with open("plots/correlations_.txt", "a") as f:
            f.write(filename + "\t" + str(corr) + "\n")
        with open("plots/error_mse_.txt", "a") as f:
            f.write(filename + "\t" + str(mse_) + "\n")
        with open("plots/error_mae_.txt", "a") as f:
            f.write(filename + "\t" + str(mea_) + "\n")
        with open("plots/r2score_.txt", "a") as f:
            f.write(filename + "\t" + str(score)+ "\n")
        """


        differences = y_selected-x_selected
        
        # Calculate mean squared error and mean absolute error
        mean = np.mean(differences)
        std  = np.std(differences)  
        
        # Histogram with the differences
        
        fig=plt.figure(figsize=(10,8))
        
        if str(kernel_label)=="cos":
            plt.hist(differences,bins=100,alpha = 0.5,color="#fde725",label="Errors") #cos color
        if str(kernel_label)=="lin":
           plt.hist(differences,bins=100,alpha = 0.5,color="#3b528b",label="Errors") #cos color
        if str(kernel_label)=="pol":      
           plt.hist(differences,bins=100,alpha = 0.5,color="#440154",label="Errors") #pol color
        if str(kernel_label)=="rbf":        
           plt.hist(differences,bins=100,alpha = 0.5, color="#5ec962",label="Errors")#rbf color
        plt.xlabel('Predidced- Real Age',fontsize = 18)
        plt.ylabel('Subjects',fontsize = 18)
        plt.ylabel('',fontsize = 20)
        plt.xlim([-100,100])
        
        #plt.title(filename.split(".")[0]+'Errors in age')
        plt.axhline(0 , color='grey', linestyle='dashed', linewidth=1)
        plt.legend(loc='upper left',fontsize = 20)

        plt.yticks(fontsize = 16)
        plt.xticks(fontsize = 16)
        # Add mean and STD to the plot
        #plt.text(0.6, 0.8, 'Mean: {:.3f}'.format(mean), transform=plt.gca().transAxes)
        #plt.text(0.6, 0.75, 'STD: {:.3f}'.format(std), transform=plt.gca().transAxes)
        plt.savefig("plots/plot_hist_"+filename.split(".")[0]+'.png', dpi=300, pad_inches = 0, format='png')
        plt.close()
                  
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_selected,differences)
        res2 = stats.spearmanr(x_selected,differences)
        corr2 =res2[0]
        corr2_ ="%.2f" % corr2
        
        fig, ax= plt.subplots(figsize=(10, 6))
       # ax.set_title("Error in age")
        
        if str(kernel_label)=="rbf":
            plt.scatter(x_selected,differences, color="#5ec962",alpha = 0.5,label="Combined MMD Kernel "+str(kernel_label)+"\n r= "+str(corr2_)+"\nRMSE: "+str(mse__)+" MEA: "+str(mea__))#rbf
        if str(kernel_label)=="cos":
            plt.scatter(x_selected, differences, color="#fde725",alpha = 0.5,label="Combined MMD Kernel "+str(kernel_label)+"\n r= "+str(corr2_)+"\nRMSE: "+"\nRMSE: "+str(mse__)+" MEA: "+str(mea__))#cos
        if str(kernel_label)=="lin":
            plt.scatter(x_selected, differences, color="#3b528b",alpha = 0.5,label="Combined MMD Kernel "+str(kernel_label)+"\n r= "+str(corr2_)+"\nRMSE: "+"\nRMSE: "+str(mse__)+" MEA: "+str(mea__))#lin
        if str(kernel_label)=="pol":
            plt.scatter(x_selected, differences, color="#440154",alpha = 0.5,label="Combined MMD Kernel "+str(kernel_label)+"\n r= "+str(corr2_)+"\nRMSE: "+"\nRMSE: "+str(mse__)+" MEA: "+str(mea__))#pol
      
        # Add linear fit to plot
        #plt.plot(np.array(x_selected), slope*np.array(x_selected) +intercept, 'grey',alpha=0.75, label="R spearman : "+str(corr2_))
        plt.ylim([-100,100])
        plt.xticks(np.arange(0,100, 5))
        plt.xlabel("Age",fontsize = 20)
        plt.ylabel("(Predicted - Real) age",fontsize = 20)
        plt.axhline(0 , color='grey', linestyle='dashed', linewidth=1)
        plt.legend(loc='upper right',fontsize = 18)
        figname = "plots/plot_"+filename.split(".")[0]+"_error_.png" 
        fig.tight_layout()
        plt.savefig(figname,dpi=200)
        
print("All plots created in file /plots .png")        
        
