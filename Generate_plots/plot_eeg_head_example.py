########################################################################
# C.Jarne-2023 Analysis Group D. Vidahurre  @cfin                      #
# This code reads a file containing variances for each channel.        #
# It then creates an MNE EvokedArray object using the variance values  #
# and plots a topographic map of the variances using plot_topomap      #
# function from MNE, with the color bar indicating the variance values.# 
# it saves the plot as a PNG file                                      #
########################################################################
import mne
import numpy as np
import matplotlib.pyplot as plt

# Define path to file with variances


filepath =".../output_files/predictions_best_fine_tuned/individual_kernel_RBF_age_real_predicted/"
pepe= filepath[-7:-4]
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
# create info object
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')

# set montage
montage = mne.channels.make_standard_montage('standard_1020')

positions = montage.get_positions()['ch_pos']

# list comprehension to get the x, y, z coordinates of ch_names
ch_pos = [positions[ch] for ch in ch_names]

pos_2d = [arr[:2].tolist() for arr in ch_pos]
print("positions dictionary",positions)
print("position array 3d",ch_pos)
print("position 2d", pos_2d)

pos_2d=np.array(pos_2d)

# Read variances from file
"""
variances = np.zeros(len(ch_names))
with open(filepath) as f:
    lines = f.readlines()
    for line in lines[1:]: # skip header line
        print(line)
        channel, variance = line.split('\t')
        variances[int(channel)] = float(variance)
"""        

import re

# Read variances from file
variances = np.zeros(len(ch_names))
with open(filepath) as f:
    lines = f.readlines()
    for line in lines:
        channel, variance = line.strip().split('\t')
        match = re.search(r'\d+', channel)
        if match is not None:
            channel = int(match.group())
            variances[channel] = float(variance)
            
       

# Create an "evoked" object with the variance values
evoked = mne.EvokedArray(variances.reshape(len(ch_names), 1), info)

# Create the heatmap
#vmin, vmax = np.exp(np.min(evoked.data)), np.exp(np.max(evoked.data))
vmin, vmax = (np.min(evoked.data)), (np.max(evoked.data))
# Set the head size
head_size = 0.82

# Create the heatmap
vmin, vmax = np.min(evoked.data), np.max(evoked.data)

dic_=marker_style={'markersize':20, 'marker':'o','fillstyle':'none', 'markerfacecolor':'black'}
mask = np.ones(evoked.data.shape[0], dtype='bool')

######## Figure #############
fig, ax = plt.subplots(figsize=(10, 10))
#fig.suptitle("Accuracies in age prediction using ridge regression per Channel",fontsize = 20)
#fig.suptitle("Accuracies in age prediction\n using MMD Kernel Cos Distance per Channel",fontsize = 20)
#fig.suptitle("Accuracies in age prediction\n using MMD Kernel RBF Distance per Channel",fontsize = 20)
#fig.suptitle("Accuracies in age prediction\n using MMD Kernel Pol Distance per Channel",fontsize = 20)
fig.suptitle("Accuracies in age prediction\n using MMD Kernel "+str(pepe)+" Distance per Channel",fontsize = 20)

img, _ = mne.viz.plot_topomap(evoked.data[:, 0], pos=pos_2d*head_size, cmap='viridis', size=5,ch_type='eeg', sensors=True, sphere=None,contours=4, 
outlines='head', names=ch_names, vlim=(vmin, vmax),show=False,axes=ax,mask=mask, mask_params=dic_)
cbar_max  = 1# 0.61
cbar_min  = 0
cbar_step = 0.1


cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='vertical', mappable=img, ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step))


cbar.set_label(r'$r^2$',size=20)
#clb = fig.colorbar(im)
#figname = "plots/Ridge_eeg_head.png" 
#figname = "plots/k_cos_eeg_head.png" 
#figname = "plots/k_gauss_eeg_head.png" 
#figname = "plots/k_pol_eeg_head.png" 
figname = "plots/k_lin_eeg_head_"+str(pepe)+".png"

fig.tight_layout()
plt.savefig(figname,dpi=200)
plt.close()

