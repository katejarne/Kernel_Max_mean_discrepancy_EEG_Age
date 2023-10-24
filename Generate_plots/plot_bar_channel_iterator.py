########################################################################################
#              C. Jarne 2023 Analysis Group of D. Vidahurre  @cfin                     #
#                                                                                      #
# This Python script conducts an analysis and visualization of EEG r2 KMER data.       #
# It begins by importing libraries for data manipulation, numerical operations,        #
# plotting, and EEG processing. Then specifies directories where specific types        #
# of files (in this case, .txt files) are located. It iterates through these           #
# directories, processing data from the selected .txt files. The data is organized,    #
# sorted, and used to generate bar plots for visualization, subsequently saved as      #
# PNG files. The script continues processing the data to create topographical maps,    #
# which are also saved as PNG files.                                                   #
# input: dir with accuracy files ( be aware of file names)                             #
# output:                                                                              #
# Accuracy_channels_kernel_+..+png                                                     #
# k_eeg_head_kernel_+..+.png                                                           #
########################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Directories where the .txt files are located

directories = ["results-predictions/KMER/results/kernel_rbf/",
               "results-predictions/KMER/results/kernel_lin/",
               "results-predictions/KMER/results/kernel_pol/"]

colores = ["#5ec962", "#3b528b", "#440154"]

# Iterate over the directories
j = 0

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
figh, axesh = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
for i, (directorio, color) in enumerate(zip(directories, colores)):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    axh = axesh[row, col]
    ax.set(frame_on=False)

    j = j+1
    # Lists to store r^2 and r^2 error values
    valores_r2 = []
    valores_incerteza = []
    etiquetas_num = []

    # files in dir
    for archivo in os.listdir(directorio):
        if archivo.startswith("KRR_r2_com") and archivo.endswith(".txt"):  # change depending on method files name
            # Extract the numerical label from the file name
            etiqueta_num = int(archivo.split("matrix")[1].split("_")[0])
            etiquetas_num.append(etiqueta_num)

            # Open the file and read the content
            with open(os.path.join(directorio, archivo), "r") as f:
                contenido = f.readlines()

            # Open the file and read the content
            for n, linea in enumerate(contenido):
                if n == 0 and linea.startswith("r^2"):
                    valor_r2 = float(linea.split("r^2")[1].strip())
                    valores_r2.append(valor_r2)
                elif linea.startswith("r^2 error 0"):
                    incerteza = float(linea.split("r^2 error")[1].strip())
                    valores_incerteza.append(incerteza)

    # Convert lists to NumPy arrays
    valores_r2 = np.array(valores_r2)
    valores_incerteza = np.array(valores_incerteza)
    etiquetas_num = np.array(etiquetas_num)
    # Sort numerical labels and corresponding values
    indices_ordenados = np.argsort(etiquetas_num)
    print(indices_ordenados)
    etiquetas_num_ordenadas = etiquetas_num[indices_ordenados]
    valores_r2_ordenados = valores_r2[indices_ordenados]
    valores_incerteza_ordenados = valores_incerteza[indices_ordenados]

    # Create the bar plot
    fig1 = plt.figure(figsize=(10, 6))
    ch_names_ = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
                 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']

    channels_ = np.arange(len(etiquetas_num_ordenadas))
    print(channels_)
    etiquetas_str = [ch_names_[i] for i in channels_]

    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(0.2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(0.1, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(0.3, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(0.4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(0.5, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(0.6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)

    barras = plt.bar(channels_, valores_r2_ordenados, color=colores[j-1], alpha=0.5, width=0.8)
    ax.bar(channels_, valores_r2_ordenados, color=colores[j-1], alpha=0.5)

    nombre_archivo = os.path.basename(os.path.normpath(directorio))
    print(nombre_archivo)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Add uncertainties
    for i, barra in enumerate(barras):
        incerteza = valores_incerteza_ordenados[i]
        x = barra.get_x() + barra.get_width() / 2
        y_top = barra.get_height() + incerteza
        plt.vlines(x, barra.get_height(), y_top, colors=color)
        plt.plot([x - 0.03, x + 0.03], [y_top, y_top], color=color, alpha=0.5)
        ax.vlines(x, barra.get_height(), y_top, colors=color)
        ax.plot([x - 0.03, x + 0.03], [y_top, y_top], color=color, alpha=0.5)

    ax.set_xlim([etiquetas_num_ordenadas[0] - 1, etiquetas_num_ordenadas[-1] + 1])
    plt.xlabel("Channel", fontsize=20)
    plt.ylabel("$R^2$", fontsize=20)
    plt.ylim([0, 0.7])
    plt.yticks(np.arange(0.0, 0.65, 0.1), fontsize = 15)

    plt.title("Accuracy KMER "+str(nombre_archivo[-2:])+" Kernel", fontsize=20, y=1.0, pad=-22)
    plt.xticks(channels_, etiquetas_str, ha="right")

    # Guardar el gráfico en un archivo PNG con el nombre correspondiente
    plt.savefig(f"plots/Accuracy_channels_{nombre_archivo}.png", bbox_inches="tight")
    plt.close(fig1)
    pepe = list(valores_r2_ordenados)
    variances = valores_r2_ordenados
    print(variances)
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
                'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']
    # create info object
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')

    # set montage
    head_size = 0.82
    montage = mne.channels.make_standard_montage('standard_1020')

    positions = montage.get_positions()['ch_pos']
    pos = montage.get_positions()['ch_pos']
    # list comprehension to get the x, y, z coordinates of ch_names
    ch_pos = [positions[ch] for ch in ch_names]

    pos_2d = [arr[:2].tolist() for arr in ch_pos]
    print("positions dictionary", positions, "\n position array 3d", ch_pos, "\n position 2d", pos_2d)

    pos_2d=np.array(pos_2d)

    # Create an "evoked" object with the variance values
    evoked = mne.EvokedArray(variances.reshape(len(ch_names), 1), info)

    # Create the heatmap
    vmin, vmax = (np.min(evoked.data)), (np.max(evoked.data))
    dic_ = marker_style = {'markersize': 20, 'marker':'o','fillstyle':'none', 'markerfacecolor':'black'}
    mask = np.ones(evoked.data.shape[0], dtype='bool')

    # now only keep the positions of the channels we want
    pos = {key: pos[key] for key in ch_names}
    # make a new montage with only the channels we are interested in
    new_pos = {key:  head_size * array for key, array in pos.items()}

    montage_true = mne.channels.make_dig_montage(ch_pos=new_pos, coord_frame='head')
    # create info object
    info = mne.create_info(ch_names=ch_names, ch_types='eeg', sfreq=1000)
    # we need to specify the sampling frequency but it is not used so it doesn't matter what we put here
    # add montage to info object - using the info object we do not need to project our 3D sensor locations
    # to a 2D plane as mne will do this for us when plotting
    info.set_montage(montage_true)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    axh.set_title("Accuracy MMD Kernel "+str(nombre_archivo), fontsize=25)
    fig2.suptitle("Accuracy KMER "+str(nombre_archivo[-3:])+" Kernel", fontsize=20)

    img, _ = mne.viz.plot_topomap(evoked.data[:,0], info, cmap='viridis', size=5, ch_type='eeg', sensors=True,
                                  sphere=None, contours=4, outlines='head', names=ch_names, vlim=(vmin, vmax),
                                  show=False, axes=ax2, mask=mask, mask_params=dic_)

    img, _ = mne.viz.plot_topomap(evoked.data[:,0], info, cmap='viridis', size=5, ch_type='eeg', sensors=True,
                                  sphere=None, contours=4, outlines='head', names=ch_names, vlim=(vmin, vmax),
                                  show=False, axes=axh, mask=mask, mask_params=dic_)

    cbar_max = 1
    cbar_min = 0
    cbar_step = 0.1

    divider = make_axes_locatable(ax2)

    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(ax=ax2, shrink=0.7, orientation='vertical', mappable=img,
                        ticks=[0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], cax=cax)
    cbar.set_label(r'$R^2$', size=20)

    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(20)
    cbar.set_label(" ",size=20)

    figname = "plots/k_eeg_head_"+str(nombre_archivo)+".png"
    fig2.tight_layout()
    plt.savefig(figname,dpi=200)
    plt.close()

    ax.set_xlabel("Channel", fontsize=22)
    ax.set_ylabel("$R^2$", fontsize=22)
    ax.set_title("Accuracy " + str(nombre_archivo) + " Kernel", fontsize=22, y=1.0, pad=-42)
    ax.set_ylim([0, 0.55])
    ax.set_xticklabels(etiquetas_str, ha="right", fontsize=12)
    ax.set_ylim(0, 0.8)

# fig.savefig("plots/all_barplots.png", dpi=200, bbox_inches="tight")
plt.tight_layout()

cbar_ax = figh.add_axes([0.9, 0.15, 0.05, 0.7])
figh.colorbar(img, cax=cbar_ax)
# figh.savefig("plots/all_head.png", dpi=200,bbox_inches="tight")
