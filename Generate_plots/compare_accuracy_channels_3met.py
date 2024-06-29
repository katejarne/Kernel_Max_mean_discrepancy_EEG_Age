
###############################################################################################
#              C. Jarne 2023 Analysis Group of D. Vidahurre  @cfin                            #
#                                                                                             #
# Code is used to processing and visualization of R2 from the 3 machine learning models:      #
# KMER, KRR and RR. It also includes statistical analysis and graph generation for comparing  #
# different configurations or approaches.                                                     #
# input: dir with accuracy files ( be aware of file names)                                    #
# output: Plot bar figures                                                                    #
###############################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as mticker

# Dir with files.txt

dirs_KMER = ["Results_files/KMER_gamma_free_0.1-1-10/"]
dirs_BR = "Results_files/BR/"
dirs_KRR = "Results_files/KRR_gamma_free_0.1-1-10/"

colores = ["grey", "#5ec962", "#5ec962", "#5ec962", "#5ec962"]
kernels = ["rbf", "lin", "pol"]

j = 0

valores_r2 = []
etiquetas_num_ = []
valores_incerteza_2 = []

for archivo in os.listdir(dirs_BR):
    if archivo.startswith("BR_r2_") and not archivo.startswith("BR_r2_folds") and archivo.endswith(".txt"):
        # Extraer la etiqueta numérica del nombre del archivo
        nombre_sin_extension = archivo.split(".")[0]
        partes = nombre_sin_extension.split("_")
        etiqueta_num = int(partes[-1])
        etiquetas_num_.append(float(etiqueta_num))
        print(etiqueta_num)

        with open(os.path.join(dirs_BR, archivo), "r") as f:
            contenido = f.readlines()

        for n, linea in enumerate(contenido):
            if n == 0 and linea.startswith("r^2"):
                valor_r2 = float(linea.split("r^2")[1].strip())
                valores_r2.append(valor_r2)
                print(valor_r2)
                incerteza = 0.00 # float(linea.split("r^2 error")[1].strip())
                valores_incerteza_2.append(incerteza)

# Convert to NumPy arrays
valores_r2 = np.array(valores_r2)

etiquetas_num = np.array(etiquetas_num_)
valores_incerteza_2 = np.array(valores_incerteza_2)

# ordering numerical labels
indices_ordenados_ = np.argsort(etiquetas_num)
etiquetas_num_ordenadas_ = etiquetas_num[indices_ordenados_]
valores_r2_ordenados_ = valores_r2[indices_ordenados_]
valores_incerteza_ordenados_ = valores_incerteza_2[indices_ordenados_]

print("R2 ordered values KRR", valores_r2_ordenados_)

ch_names_ = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
             'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']
channels_ = np.arange(len(etiquetas_num_ordenadas_))
etiquetas_str = [ch_names_[i] for i in channels_]

etiquetas_num_3_ = []
valores_r2_3 = []
valores_incerteza_3 = []

for archivo in os.listdir(dirs_KRR):
    if archivo.startswith("KRR_r2_com") and not archivo.startswith("KRR_r2_folds") and archivo.endswith(".txt"):

        nombre_sin_extension = archivo.split(".")[0]
        partes = nombre_sin_extension.split("_")
        etiqueta_num = int(partes[-1])
        etiquetas_num_3_.append(float(etiqueta_num))
        print(etiqueta_num)

        with open(os.path.join(dirs_KRR, archivo), "r") as f:
            contenido = f.readlines()

        for n, linea in enumerate(contenido):
            if n == 0 and linea.startswith("r^2"):
                valor_r2 = float(linea.split("r^2")[1].strip())
                valores_r2_3.append(valor_r2)
                incerteza = 0.00
                valores_incerteza_3.append(incerteza)

valores_r2_3 = np.array(valores_r2_3)
etiquetas_num_3 = np.array(etiquetas_num_3_)
valores_incerteza_3 = np.array(valores_incerteza_3)

indices_ordenados_3 = np.argsort(etiquetas_num_3)
etiquetas_num_ordenadas_3 = etiquetas_num_3[indices_ordenados_3]
valores_r2_ordenados_3 = valores_r2_3[indices_ordenados_3]
valores_incerteza_ordenados_3 = valores_incerteza_3[indices_ordenados_3]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

for i, directorio_error in enumerate(dirs_KMER):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    ax.set(frame_on=False)

    valores_error = []
    etiquetas_num_KMER = []
    valores_incerteza = []

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set(frame_on=False)
    for archivo in os.listdir(directorio_error):

        #if archivo.startswith("KRR_r2_com") and archivo.endswith(".txt"):
        if archivo.startswith("KMER_r2") and not archivo.startswith("KMER_r2_folds") and archivo.endswith(".txt"):
            print(archivo)
            etiqueta_num = int(archivo.split("KMER_r2_com_")[1].split(".")[0])

            print(etiqueta_num)
            etiquetas_num_KMER.append(etiqueta_num)

            with open(os.path.join(directorio_error, archivo), "r") as f:
                contenido = f.readlines()

            for n, linea in enumerate(contenido):
                if n == 0 and linea.startswith("r^2"):
                    valor_r2 = float(linea.split("r^2")[1].strip())
                    valores_error.append(valor_r2)

                    incerteza = 0.00
                    valores_incerteza.append(incerteza)

    # Convert to numpy arrays
    valores_error = np.array(valores_error)
    etiquetas_num_KMER = np.array(etiquetas_num_KMER)
    valores_incerteza = np.array(valores_incerteza)
    indices_ordenados = np.argsort(etiquetas_num_KMER)

    etiquetas_num_ordenadas = etiquetas_num_KMER[indices_ordenados]

    valores_error_ordenados = valores_error[indices_ordenados]
    valores_incerteza_ordenados = valores_incerteza[indices_ordenados]
    posiciones = np.arange(len(etiquetas_num_ordenadas))
    barras = ax.bar(posiciones, valores_error_ordenados, color=colores[i+1], alpha=0.5, label="KMER")
    ax2.bar(posiciones+0.5, valores_error_ordenados,width=0.3, color=colores[i+1], alpha=0.5, label="KMER ")

    barras_3 = ax.bar(posiciones, valores_r2_ordenados_3, color="pink", alpha=0.7)
    ax2.bar(posiciones+0.2, valores_r2_ordenados_3, width=0.3, color="pink", alpha=0.7, label="KRR")

    for jj, barra in enumerate(barras_3):
        incerteza = valores_incerteza_ordenados_3[jj]
        x = barra.get_x() + barra.get_width() / 2
        y_top = barra.get_height() + incerteza
        ax.vlines(x, barra.get_height(), y_top, colors="pink")
        ax.plot([x+0.12 - 0.03, x+0.12 + 0.03], [y_top, y_top], color="pink", alpha=0.5)
        ax2.vlines(x+0.2, barra.get_height(), y_top, colors="pink")

    for jj, barra in enumerate(barras):
        incerteza = valores_incerteza_ordenados[jj]
        x = barra.get_x() + barra.get_width() / 2
        y_top = barra.get_height() + incerteza
        ax.vlines(x, barra.get_height(), y_top, colors=colores[i+1])
        ax.plot([x+0.5 - 0.03, x+0.5 + 0.03], [y_top, y_top], color=colores[i+1], alpha=0.5)
        ax2.vlines(x+0.5, barra.get_height(), y_top, colors=colores[i+1])

    barras_2 = ax.bar(posiciones, valores_r2_ordenados_, color=colores[0], alpha=0.7)
    ax2.bar(posiciones-0.1, valores_r2_ordenados_,width=0.3, color=colores[0], alpha=0.7, label="RR")
    for jj, barra in enumerate(barras_2):
        incerteza = valores_incerteza_ordenados_[jj]
        x = barra.get_x() + barra.get_width() / 2
        y_top = barra.get_height() + incerteza
        ax.vlines(x, barra.get_height(), y_top, colors=colores[0])
        ax.plot([x+0.06 - 0.03, x+0.06 + 0.03], [y_top, y_top], color=colores[0], alpha=0.5)
        ax2.vlines(x+0.06-0.2, barra.get_height(), y_top, colors=colores[0])

    ax.set_xlabel("Channel", fontsize=12)
    ax.set_ylabel(r'$R^2$', fontsize=12)
    ax.set_xticks(posiciones)
    ax.set_xticklabels(etiquetas_str, ha="right", fontsize=13)
    #ax.set_yticks(np.arange(0, 0.66, 0.1), ha="right", fontsize=18)
    ax.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(0.2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(0.1, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(0.3, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(0.4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(0.5, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.axhline(0.6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.legend(fontsize=15)
    ax.set_yticks(np.arange(0, 0.66, 0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.set_yticklabels(np.arange(0, 0.61, 0.1), ha="right", fontsize=18)

    ax2.set_xlabel("Channel", fontsize=12)
    ax2.set_ylabel(r'$R^2$', fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax2.set_xticks(posiciones)
    ax2.set_xticklabels(etiquetas_str, ha="left", fontsize=12)
    ax2.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax2.axhline(0.1, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax2.axhline(0.2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax2.axhline(0.3, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax2.axhline(0.4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax2.axhline(0.5, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax2.axhline(0.6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    ax2.legend(fontsize=12)
    print(np.arange(0, 0.66, 0.1))
    ax2.set_yticks(np.arange(0, 0.66, 0.1))
    ax2.set_yticklabels(np.arange(0, 0.66, 0.1), ha="right", fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    # Guardar la figura individual
    plt.tight_layout()
    plt.savefig(f"plots/{kernels[i]}.png", dpi=200)
    plt.close(fig2)  # Cerrar la figura actual para liberar memoria

    # Statistical test
    p_values = []
    diference = valores_error_ordenados-valores_r2_ordenados_#valores_r2_ordenados_3
    res = stats.wilcoxon(diference, alternative='greater')
    print(kernels[i], "Test result ", res)
    print("---------------")

    performance_diff = [kmer - br for kmer, br in zip(valores_error_ordenados, valores_r2_ordenados_3)]

    # Plot of the differences
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 4))
    ax2.bar(posiciones, performance_diff, color='green', alpha=0.7)
    ax2.set_xticks(posiciones)
    ax2.set_xticklabels(etiquetas_str, ha="left", fontsize=12)
    ax2.set_xlabel('Channels')
    ax2.set_ylabel('Differences')
    ax2.set_title('Differences KMER and KRR')
    plt.tight_layout()
    plt.savefig(f"plots/diff_{kernels[i]}.png", dpi=200)
    plt.close(fig2)

print("All plots created in file /plots .png")
