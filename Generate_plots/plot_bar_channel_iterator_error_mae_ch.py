##################################################################################
#       C. Jarne 2023 Analysis Group of D. Vidahurre  @cfin                      #
#                                                                                #
#  This script conducts a visualization of Mean Absolute Error (MAE)             #
#  data across different EEG channels. It begins by importing necessary          #
#  libraries for data manipulation, numerical operations, and plotting.          #
#  The script then specifies directories where specific types of files           #
#  (in this case, error-related .txt files) are located and assigns colour       #
#  codes for visualization. Finally, the resulting figures are saved, and        #
#  there's an option to save an aggregated figure containing all the bar plots.  #
#                                                                                #
# input: dir with accuracy files ( be aware of file names)                       #
# output: error_mae_channels_+...+.png                                           #
##################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

directories = ["results-predictions/KMER/results/errores_scores/rbf/",
               "results-predictions/KMER/results/errores_scores/lin/",
               "results-predictions/KMER/results/errores_scores/pol/"]

# Colors
colores =["#5ec962", "#3b528b", "#440154"]
# colores = cm.viridis_r(np.linspace(0, 1, 3))

j = 0
# Iterate over the directories

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
for i, (directorio, color) in enumerate(zip(directories, colores)):
    print(directorio)
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    ax.set(frame_on=False)
    j = j+1

    valores_r2 = []
    valores_incerteza = []
    etiquetas_num = []

    # files in dir
    for archivo in os.listdir(directorio):
        if archivo.startswith("error_mae_") and archivo.endswith(".txt"):
            # Abrir el archivo y leer el contenido
            with open(os.path.join(directorio, archivo), "r") as f:
                contenido = f.readlines()

            # Buscar los valores de r^2 y r^2 error en el contenido
            for n, linea in enumerate(contenido):
                if linea.startswith("predicciones_eeg"):
                    etiqueta_num = int(linea.split("matrix")[1].split("_")[0])
                    etiquetas_num.append(etiqueta_num)
                    valor_r2 = float(linea.split("\t")[1].strip())
                    valores_r2.append(valor_r2)

    valores_r2 = np.array(valores_r2)
    valores_incerteza = np.array(valores_incerteza)
    etiquetas_num = np.array(etiquetas_num)

    indices_ordenados = np.argsort(etiquetas_num)
    etiquetas_num_ordenadas = etiquetas_num[indices_ordenados]
    valores_r2_ordenados = valores_r2[indices_ordenados]

    fig1 = plt.figure(figsize=(10, 6))
    ch_names_ = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']

    channels_=np.arange(len(etiquetas_num_ordenadas))
    etiquetas_str = [ch_names_[i] for i in channels_]

    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(8, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(10, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(12, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)

    barras = plt.bar(channels_, valores_r2_ordenados, color=colores[j-1], alpha=0.5)
    ax.bar(channels_, valores_r2_ordenados, color=colores[j-1], alpha=0.5)

    nombre_archivo = directorio[-4:-1]
    print(nombre_archivo)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xlabel("Channel", fontsize=20)
    plt.ylabel("MAE [years]", fontsize=20)
    plt.title("Error per Channel", fontsize=20, y=1.0, pad=-24)
    plt.xticks(channels_, etiquetas_str, ha="right", fontsize=17)
    plt.yticks(fontsize=17)
    plt.ylim([0, 14.5])
    plt.savefig(f"plots/error_mae_channels_{nombre_archivo}.png", bbox_inches="tight")
    plt.close(fig1)

    ax.set_xlabel("Channel", fontsize=12)
    ax.set_ylabel("MAE [years]", fontsize=12)
    ax.set_title("Error " + str(nombre_archivo) + " Kernel", fontsize=12, y=1.0, pad=-8)
    ax.set_xticks(channels_)
    ax.set_yticks(np.arange(0.0, 12.1,2), fontsize=15)
    ax.set_xticklabels(etiquetas_str, ha="right", fontsize=12)

    ax.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(8, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(10, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(12, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)

plt.tight_layout()
# fig.savefig("plots/all_barplots_error_mae.png", dpi=200,bbox_inches="tight")


