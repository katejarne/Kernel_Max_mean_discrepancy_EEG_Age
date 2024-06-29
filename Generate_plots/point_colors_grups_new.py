import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Directory where the files are located
directory = "Results_files/KMER_gamma_free_0.1-1-10/"
# Get all files matching the pattern
files = glob.glob(os.path.join(directory, "KMER_prediction_*.txt"))

feature_file_path = "...Feature_file/features_ok.txt"
features_df = pd.read_csv(feature_file_path, delimiter='\t', header=None)
ordered_categories = features_df[1].unique().tolist()

# Function to generate a unique color for each category


def generate_colors(num_colors):
    colors = plt.cm.get_cmap('tab20b', num_colors)
    return [colors(i) for i in range(num_colors)]


def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"File not found: {filename}")

# Dic

category_results = {category: [] for category in ordered_categories}

indices = []
size_cat = []
range_age_list = []
for file in files:
    file_name = os.path.basename(file)
    number = int(extract_number(file_name))
    print(number)
    indices.append(number)
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file, delimiter='\t', header=None)
    df.columns = ['Col1', 'Col2', 'Col3', 'Col4']

    df['Col1'] = df['Col1']  # np.exp(df['Col1'])
    df['Col2'] = df['Col2']  # np.exp(df['Col2'])

    colors = generate_colors(len(ordered_categories))
    color_map = dict(zip(ordered_categories, colors))

    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'aspect': 0.65})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    lista_mea = []
    lista_r2 = []
    lista_size = []
    for category in ordered_categories:
        subset = df[df['Col4'] == category]
        mse_ = mean_squared_error(subset['Col1'], subset['Col2'], squared=False)
        mea_ = mean_absolute_error(subset['Col1'], subset['Col2'])
        r2 = r2_score(subset['Col1'], subset['Col2'])
        size = len(subset['Col1'])
        if number == 1:
            size_cat.append(size)

        # Save MAE in dic
        category_results[category].append(mea_)

        lista_mea.append(mea_)
        lista_r2.append(r2)
        lista_size.append(size)

        plt.scatter(subset['Col1'], subset['Col2'], label=category, color=color_map[category], alpha=0.5)
    xx = np.arange(0, 98, 1)
    #yy = slope * xx + intercept
    yy = xx
    plt.plot(xx, yy,"--", color="gray")
    plt.ylim([-10, 101])
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1)
    plt.xlim([0, 100])
    plt.xlabel('Age', fontsize=20)
    plt.ylabel('Predicted age', fontsize=20)
    plt.yticks(np.arange(0.0, 101, 20), fontsize=17)
    plt.xticks(np.arange(0.0, 101, 5), fontsize=17)
    ax.legend(loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=18)
    plt.savefig(os.path.join("plots", f'{os.path.basename(file)}_plot.png'))
    plt.close()

    fig = plt.figure(figsize=(12, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=-65)
    ax.grid(False)
    marker = [".", ",", "o", "v", "^", "D", ">", "s", "p", "P", "*",
              "h", "H", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for i, category in enumerate(ordered_categories):
        subset = df[df['Col4'] == category]
        ax.scatter(np.zeros_like(subset["Col2"]) + i * 2, subset['Col1'], subset['Col2'],
                   label=category,
                   color=color_map[category], s=50, alpha=0.75, marker=marker[i])

    xx_ = np.arange(0, 100, 1)
    yy_ = xx_
    ax.set_ylim([-10, 101])
    ax.set_xlim([0, 27])
    ax.set_ylabel('Age', fontsize=20, labelpad=20)
    ax.set_xlabel('Batch', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0.0, 101, 20), fontsize=36)
    ax.set_zticks(np.arange(-20.0, 101, 20), fontsize=36)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    plt.savefig(os.path.join("plots", f'{os.path.basename(file)}_plot_3d.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Violin plot for the difference between Col2 and Col1
    fig, ax = plt.subplots(figsize=(14, 6.5))
    df['Col2_Col1_Diff'] = df['Col2'] - df['Col1']
    data_diff = [df[df['Col4'] == category]['Col2_Col1_Diff'] for category in ordered_categories]
    parts_diff = ax.violinplot(data_diff, showmeans=False, showmedians=False, showextrema=False)

    for pc, category in zip(parts_diff['bodies'], ordered_categories):
        pc.set_facecolor(color_map[category])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    means_diff = [np.mean(d) for d in data_diff]
    for i, mean in enumerate(means_diff):
        ax.plot(i + 1, mean, 'wo', markersize=8, markeredgecolor='black')
    ax.set_xticks(np.arange(1, len(ordered_categories) + 1))
    ax.set_xticklabels(ordered_categories, rotation=30, fontsize=17, ha="center")
    ax.set_ylabel('Delta (Predicted - actual age)', fontsize=18)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_yticks(np.arange(-60,50,10))
    ax.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    fig.tight_layout()
    violin_plot_path_diff = os.path.join("plots", f'{os.path.basename(file)}_violin_plot_diff.png')
    plt.savefig(violin_plot_path_diff, dpi=200, pad_inches=0, transparent=False, format='png')
    plt.close()

    # Violin plot for age distribution
    fig, ax = plt.subplots(figsize=(14, 6))
    data = [df[df['Col4'] == category]['Col1'] for category in ordered_categories]
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    for pc, category in zip(parts['bodies'], ordered_categories):
        pc.set_facecolor(color_map[category])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    means = [np.mean(d) for d in data]
    rmss = [np.std(d) for d in data]
    for i, mean in enumerate(means):
        ax.plot(i + 1, mean, 'wo', markersize=8, markeredgecolor='black')
        ax.plot([i + 1 - 0.1, i + 1 + 0.1], [mean + rmss[i], mean + rmss[i]], color='black', lw=1)
        ax.plot([i + 1 - 0.1, i + 1 + 0.1], [mean - rmss[i], mean - rmss[i]], color='black', lw=1)
        ax.plot([i + 1, i + 1], [mean - rmss[i], mean + rmss[i]], color='black', lw=1)

    ax.set_xticks(np.arange(1, len(ordered_categories) + 1))
    ax.set_xticklabels(ordered_categories, rotation=30, fontsize=17, ha="center")
    ax.set_ylabel('Age', fontsize=18)
    ax.tick_params(axis='y', labelsize=17)
    ax.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    violin_plot_path = os.path.join("plots", f'{os.path.basename(file)}_violin_plot_age_dist.png')
    fig.tight_layout()
    plt.savefig(violin_plot_path, dpi=200, pad_inches=0, transparent=False, format='png')
    plt.close()

    with open(f"plots/r2_site{file_name}_.txt", "w") as file:
            file.write("site\t\MAE\tR2\tsize\n")
            for i, element in enumerate(ordered_categories):
                file.write(f"{element}\t")
                file.write(f"{lista_mea[i]}\t")
                file.write(f"{lista_r2[i]}\t")
                file.write(f"{lista_size[i]}\n")

    if number==1:
        max_min_values = {category: (max(values), min(values)) for category, values in zip(ordered_categories, data)}
        for category, (max_val, min_val) in max_min_values.items():
            range_age_list.append([int(min_val),int(max_val)])
print("Scatter plots have been created for all files.")

# Save category_results to a file if needed
with open("plots/category_mea_results.txt", "w") as f:
    for category, mea_list in category_results.items():
        f.write(f"{category}: {mea_list}\n")

def head_plot(variances,site):
    print(variances)
    variances = np.array(variances)
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
    vmin, vmax = 0, 30 #(np.min(evoked.data)), (np.max(evoked.data))
    dic_ = marker_style = {'markersize': 20, 'marker':'o','fillstyle':'none', 'markerfacecolor':'black'}
    mask = np.ones(evoked.data.shape[0], dtype='bool')

    # now only keep the positions of the channels we want
    pos = {key: pos[key] for key in ch_names}
    # make a new montage with only the channels we are interested in
    new_pos = {key:  head_size * array for key, array in pos.items()}

    montage_true = mne.channels.make_dig_montage(ch_pos=new_pos, coord_frame='head')
    # create info object
    info = mne.create_info(ch_names=ch_names, ch_types='eeg', sfreq=1000)
    # we need to specify the sampling frequency, but it is not used and it doesn't matter what we put here
    # add montage to info object - using the info object we do not need to project our 3D sensor locations
    # to a 2D plane as mne will do this for us when plotting
    info.set_montage(montage_true)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.set_title(f"{category}\n Subjects: {sujetos} range: {rango} years", fontsize=25)
    # fig2.suptitle("Accuracy KMER "+str(nombre_archivo[-3:])+" Kernel", fontsize=20)

    img, _ = mne.viz.plot_topomap(evoked.data[:, 0], info, cmap='jet', size=5, ch_type='eeg', sensors=True,
                                  sphere=None, contours=4, outlines='head', names=ch_names, vlim=(vmin, vmax),
                                  show=False, axes=ax2, mask=mask, mask_params=dic_)

    cbar_max = 1
    cbar_min = 0
    cbar_step = 0.1

    divider = make_axes_locatable(ax2)

    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(ax=ax2, shrink=0.7, orientation='vertical', mappable=img, cax=cax)
    cbar.set_label(r'$MAE$', size=20)

    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(20)
    cbar.set_label(r'MAE',size=20)

    figname = f"plots/k_eeg_head_{site}.png"
    fig2.tight_layout()
    plt.savefig(figname,dpi=200)
    plt.close()

indices_ordenados = np.argsort(indices)
order_index_ordered = np.array(indices)[indices_ordenados]

j=0
for category, mea_list in category_results.items():
    mea_ordered = np.array(mea_list)[indices_ordenados]
    sujetos = size_cat[j]
    rango = range_age_list[j]
    pepe = head_plot(mea_ordered,category)
    j = j+1

