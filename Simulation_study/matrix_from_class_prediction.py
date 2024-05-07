###########################################################
#  C. Jarne 2024 Analysis Group of D. Vidahurre  @cfin    #
# This code generates the matrix with the accuracy values #
# for each point of the grid that we have created with    #
# the simulations, for which we have made the predictions #
# saved in output_files. Predictions are generated from   #
# using KMER_for_syntetic.py or KRR_for_syntetic.py       #
###########################################################
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Dir
directorio = "output_files"

# sizes and values
# sizes = [50, 100, 200, 300, 400, 500]
sizes = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
#alpha2_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
alpha2_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]


matriz_valores = np.zeros((len(alpha2_values), len(sizes)))

# Loop over directories
for nombre_archivo in os.listdir(directorio):
    if nombre_archivo.startswith("KMER_r2_alpha2") and nombre_archivo.endswith("_distances.txt"): # kmer
    #if nombre_archivo.startswith("KRR_r2_com_alpha2_") and nombre_archivo.endswith(".txt"):
        # Extraer los valores alpha2 y size del nombre del archivo
        alpha2_valor = float(re.search(r'alpha2_(\d+\.\d+)', nombre_archivo).group(1))
        size_valor = int(re.search(r'size_(\d+)_distances.txt', nombre_archivo).group(1)) #kmer
        # size_valor = int(re.search(r'size_(\d+).txt', nombre_archivo).group(1)) # krr

        # Leer el valor r^2 del archivo
        with open(os.path.join(directorio, nombre_archivo), 'r') as archivo:
            contenido = archivo.readlines()
            for linea in contenido:
                if linea.startswith("r^2"):
                    valor_r2 = float(linea.split("r^2")[1].strip())

                    # Buscar la posición correspondiente en la matriz y asignar el valor r^2
                    alpha2_index = alpha2_values.index(alpha2_valor)
                    size_index = sizes.index(size_valor)
                    matriz_valores[alpha2_index, size_index] = valor_r2

#tick_label_y = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
#tick_label_x = [50, 100, 200, 300, 400, 500]
tick_label_y = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
tick_label_x = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]

#np.savetxt("matriz_valores_KRR.txt", matriz_valores, fmt='%.6f')
np.savetxt("matriz_valores_KMER.txt", matriz_valores, fmt='%.6f')
print("matrix saved in txt file'")
fig, ax = plt.subplots(figsize=(10, 6.5))
plt.imshow(matriz_valores, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
#plt.colorbar(label='Accuracy KMER')
#plt.colorbar(label=r'$R^2$ KRR')
plt.colorbar(label=r'$R^2$ KMER')
ax.invert_yaxis()
plt.xticks(np.arange(len(matriz_valores[0])), tick_label_x)
#plt.yticks(np.arange(len(matriz_valores[1])), tick_label_y)
plt.yticks(np.arange(len(tick_label_y)), tick_label_y)
plt.xlabel('Samples in Histogram')
plt.ylabel('Difference between alpha parameters')
plt.title(r'$R^2$ Matrix')
plt.tight_layout()
plt.savefig("accuracy_matrix_KMER_predi.png", dpi=200)
#plt.savefig("accuracy_matrix_KRR_predi.png", dpi=200)
plt.close(fig)
