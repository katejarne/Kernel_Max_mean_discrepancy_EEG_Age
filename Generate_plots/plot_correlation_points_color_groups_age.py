###############################################################################
# C.Jarne- 2023 Analysis Group D. Vidahurre  @cfin                            #
# Code to generate  scatterplots of the actual ages vs. the predicted ages    #  
# where the points will be colored according sites.                           # 
# In addition, a linear fit line is added and the Pearson correlation         #      
# value is displayed in the figure caption.                                   # 
# The figure is saved in .png format.                                         #
###############################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace
from matplotlib import cm
from scipy.stats import linregress
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Directory containing files (matrices) to be analyzed (combined or individual)

dir_path=".../output_files/predictions_best_fine_tuned/combined_kernel_age_real_predict/"

# Initialize list to hold all histogram data
all_hist_data = []


# Iterate over all files in the directory
for filename in os.listdir(dir_path):
    if not filename.endswith('.txt'):
        continue  # Skip non-text files
    pepe= os.path.splitext(filename)[0]
    pepe=pepe[-3:]
    # read in data from first table
    table1 = np.loadtxt(os.path.join(dir_path, filename), dtype=float)
    data=table1.T
    x = data[0]
    y = data[1]

    # Calculate linear regression and correlation
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    res = stats.spearmanr(x,y)
    print("Correlation ",res[0])
    corr  =res[0]
    corr_ ="%.2f" % corr
        
    # Calculate mean squared error and mean absolute error
    mse_ =mean_squared_error(x, y, squared=False)
    mea_ = mean_absolute_error(x, y)
    mse1__="%.2f" % mse_
    mea1__="%.2f" % mea_


    # create a dictionary to store data from second table
    groups       = {}
    num_elements = {}
    
    # read in data from second table (Fetures file)
    with open(os.path.join('.../input_files/Feature_file/salida_features_ok.txt'), 'r') as f:
        for k,line in enumerate(f):
            row = line.strip().split('\t')
            if row[1] not in groups:
                groups[row[1]] = {'x': [], 'y': []}
                num_elements[row[1]] = 0
            if row[2] != 'nan' and k<1926: #1949:                         
                groups[row[1]]['x'].append(float(table1[k][0]))
                groups[row[1]]['y'].append(float(table1[k][1]))
                num_elements[row[1]] += 1

    # create scatter plot
    #fig, ax = plt.subplots(figsize=(15,8), subplot_kw={'aspect': 0.65})
    fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'aspect': 0.65})
    num_groups    = len(groups)
    cm_subsection = linspace(0, 1, num_groups) 

    colors = [ cm.jet(x) for x in cm_subsection ]
  
    for i, (group, data) in enumerate(groups.items()):
        mse_ =mean_squared_error(data['x'], data['y'], squared=False)
        mea_= mean_absolute_error(data['x'], data['y'])
        mse__="%.2f" % mse_
        mea__="%.2f" % mea_
        #ax.scatter(data['x'], data['y'], label=group+"\nRMSE: "+str(mse__)+" MEA: "+str(mea__), color = colors[i],alpha=0.43)
        ax.scatter(data['x'], data['y'], label=group, color = colors[i],alpha=0.43)

    xx = np.arange(0, 100, 1)
    yy = xx
    plt.plot(xx, yy,'grey',alpha=0.75,label="r= "+str(corr_)+"\nRMSE: "+str(mse1__)+" MAE: "+str(mea1__))
    # Add linear fit to plot   
    #plt.plot(x, intercept + slope*x, 'grey',alpha=0.75, label="R= "+str(corr_)+" (Spearman)\nRMSE: "+str(mse1__)+" MAE: "+str(mea1__))
    ax.set_title('Age prediction with Combined MMD Kernel '+str(pepe),fontsize = 20)
    plt.ylim([-10,101])
    plt.axhline(0 , color='grey', linestyle='dashed', linewidth=1)
    plt.xlim([0,100])
   
    plt.xlabel('Age',fontsize = 20)
    plt.ylabel('Predicted age',fontsize = 20)
    
    plt.yticks(np.arange(0.0, 101,20),fontsize = 15)
    plt.xticks(np.arange(0.0, 101,5),fontsize = 15)
    
    # Legend
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12)
    plt.savefig('plots/plot_'+os.path.splitext(filename)[0]+'.png', dpi=300,bbox_inches='tight')
    plt.close()

    hist_data = []
    #fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax =plt.subplots(figsize=(10,8))
    for i, (group, data) in enumerate(groups.items()):
        diff = np.array(data['y']) - np.array(data['x'])
        
        
        # mean and std
        mean  = np.mean(diff)
        std   = np.std(diff)  
        meann ="%.2f" % mean
        stdd  ="%.2f" % std
        ax.hist(diff, bins=50, range=[diff.min() ,diff.max()], alpha=0.35, label=group+" MSE: "+str(stdd)+" Mean: "+str(meann), color = colors[i])
        
        #ax.set_title("Error  [years] "+ f'(for {k+1} individuals) Kernel '+str(pepe))
        hist_data.append(diff)
    all_hist_data.append(hist_data)
    plt.xlabel('Predidced- Real Age',fontsize = 18)
    plt.ylabel('Subjects',fontsize = 18)

    plt.xlim([-100,100])
    plt.legend(loc='upper left')
    ax.legend()
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    fig.tight_layout()
    plt.savefig(f'plots/plot_color"+"{os.path.splitext(filename)[0]}_{group}_histogram.png', dpi=200, pad_inches=0, transparent=False, format='png')
    plt.close()
                
    print("Number of elements per group:")
    print(num_elements)
print("All plots created in file /plots .png")     
    


