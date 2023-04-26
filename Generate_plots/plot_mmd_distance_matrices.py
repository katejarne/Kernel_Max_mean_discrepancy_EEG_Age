######################################################################
#  C.Jarne 2023 Analysis Group D. Vidahurre  @cfin                   #
#This code creates a figure with the title "MMD distance matrix"     #
#and plots the input matrix as a heatmap using a specified color map.# 
#It saves the figure as a .png file in a subdirectory named "plots". #
######################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.cm as cm
from pylab import grid

r_dir="/Distance_matrix_dir..."

cmap = plt.get_cmap('viridis')
 
for filename in os.listdir(r_dir):
    if filename.endswith(".txt"):
        MMD_matrix  =np.genfromtxt(r_dir+ filename, delimiter='\t')
        print("MMD Matrix",MMD_matrix)
        print("len", len(MMD_matrix[0]))
    
        ind_pol = np.unravel_index(np.argmax(MMD_matrix, axis=None), MMD_matrix.shape) 
        print("ind",ind_pol)
    
        fig=plt.figure(figsize=(17,12.5))
        grid(True)
        fig.suptitle( "MMD distance matrix",fontsize = 28)
        im=plt.imshow(MMD_matrix,interpolation='none',cmap=cmap,label='MMD matrix', aspect="auto",vmin =np.min(MMD_matrix), vmax =np.max(MMD_matrix) ,origin='lower')
        plt.colorbar(im, orientation='vertical')
        """
        plt.xticks(np.arange(0,len(MMD_matrix[0]), 1))
        plt.yticks(np.arange(0,len(MMD_matrix[0]), 1))
        plt.ylabel('Subject [i]',fontsize = 20) #Post-synaptic
        plt.xlabel('Subject [i]',fontsize = 20)
        
        """
        #plt.legend(loc='upper right')
        figname = "plots/distance_matrix_"+filename.split(".")[0]+"_.png" 
        plt.savefig(figname,dpi=200)
        plt.close()
    
    
  
