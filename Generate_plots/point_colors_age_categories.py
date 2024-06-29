import os
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Dir
directory = "Results_files/KMER_gamma_free_0.1-1-10/"

files = glob.glob(os.path.join(directory, "KMER_prediction_*.txt"))

# Leer el archivo de características y extraer las categorías
feature_file = ".../Feature_file/features_ok.txt"
caracteristicas_df = pd.read_csv(feature_file, delimiter='\t', header=None)
categorias_ordenadas = caracteristicas_df[1].unique().tolist()

# ranges
categories = ["5-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-90"]
etiquetas_categorias = {"5-10": '5-10', "10-20": '10-20', "20-30": '20-30', "30-40": '30-40', "40-50": '40-50',
                        "50-60": '50-60', "60-70": '60-70', "70-90": '70-90'}

# Aplicar mapa de colores
colores = plt.cm.viridis(np.linspace(0, 1, len(categories)))
mae_por_categoria = [[] for _ in range(len(categories))]
r2_por_categoria = [[] for _ in range(len(categories))]
nombres_canales = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                   'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']
indices_ordenados = []

def extraer_numero(nombre_archivo):
    coincidencia = re.search(r'prediction_(\d+)', nombre_archivo)
    if coincidencia:
        return int(coincidencia.group(1))
    else:
        raise ValueError(f"No se encontró un número en el nombre del archivo: {nombre_archivo}")

# Read files


for archivo in files:
    print(archivo)
    numero = extraer_numero(archivo)
    if numero == 18:
        nombre_canal = nombres_canales[-1]
    else:
        nombre_canal = nombres_canales[numero]
    print(numero)
    indices_ordenados.append(int(numero))
    print(nombre_canal)
    df = pd.read_csv(archivo, delimiter='\t', header=None)
    df.columns = ['Col1', 'Col2', 'Col3', 'Col4']

    # Filter ages
    df['Edad'] = pd.cut(df['Col1'], bins=[5, 10, 20, 30, 40, 50, 60, 70, 90], labels=categories)
    df = df[df['Edad'].isin(categories)]

    fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={'aspect': 0.65})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    r2_range_age = []
    mea_range = []
    for i, categorie in enumerate(categories):
        subset = df[df['Edad'] == categorie]
        if len(subset) > 0:
            mae = mean_absolute_error(subset['Col1'], subset['Col2'])
            r2 = r2_score(subset['Col1'], subset['Col2'])
            r2_range_age.append(r2)
            mea_range.append(mae)
            label = f"{etiquetas_categorias[categorie]}\nMEA: {mae:.2f}"
            ax.scatter(subset['Col1'], subset['Col2'], label=label, color=colores[i], alpha=0.5)

            mae_por_categoria[i].append(mae)
            r2_por_categoria[i].append(r2)

    ax.set_ylim([-10, 101])
    ax.axhline(0, color='grey', linestyle='dashed', linewidth=1)
    ax.plot([-10, 101], [-10, 101], color='gray', linestyle='--', label='y=x')

    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(np.arange(0, 101, 5))
    ax.set_xlabel('Age', fontsize=20)
    ax.set_ylabel('Predicted age', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=18)

    # Título del gráfico con el nombre del canal
    ax.set_title(f'Age for channel {nombre_canal}', fontsize=20)

    nombre_grafico = os.path.join("plots", f'{os.path.basename(archivo)}_edad_plot.png')
    plt.savefig(nombre_grafico)
    plt.close(fig)

    print(r2_range_age)

    #  MEA and R^2
    mae_promedio_por_canal = mea_range
    r2_promedio_por_canal = [np.mean(rs) for rs in r2_por_categoria]

    # Bar plot
    plt.figure(figsize=(10, 6))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.bar(np.arange(len(mae_promedio_por_canal)), mae_promedio_por_canal, color=colores, alpha=0.5)
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.xlabel('Age ranges', fontsize=16)
    plt.xticks(np.arange(len(etiquetas_categorias)), etiquetas_categorias.values(), ha="center", fontsize=17)
    plt.axhline(5, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(10, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(15, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(20, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(25, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.axhline(30, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    plt.ylabel('MEA', fontsize=16)
    plt.yticks(fontsize=17)
    plt.title(f'Mean absolute error for channel {nombre_canal}', fontsize=18)
    plt.savefig(f"plots/{os.path.basename(archivo)}_mae_promedio_por_canal.png")
    plt.close()

resultados_finales = []

for i, categorie in enumerate(categories):
    if mae_por_categoria[i]:
        mejor_canal = nombres_canales[np.argmin(mae_por_categoria[i])]
        peor_canal = nombres_canales[np.argmax(mae_por_categoria[i])]
        mejor_mae = min(mae_por_categoria[i])
        peor_mae = max(mae_por_categoria[i])
        resultados_finales.append((categorie, mejor_canal, mejor_mae, peor_canal, peor_mae))

for resultado in resultados_finales:
    categorie, mejor_canal, mejor_mae, peor_canal, peor_mae = resultado
    print(f"Para el rango de edad {categorie}:")
    print(f"  Mejor canal: {mejor_canal} con MEA {mejor_mae:.2f}")
    print(f"  Peor canal: {peor_canal} con MEA {peor_mae:.2f}")
