import os
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable

directory = "Results_files/KMER_gamma_free_0.1-1-10/"
# Get all files matching the pattern
files = glob.glob(os.path.join(directory, "KMER_prediction_*.txt"))
#files = glob.glob(os.path.join(directory, "KRR_prediction_*.txt"))

# Feature file
feature_file_path = ".../Feature_file/features_ok.txt"
features_df = pd.read_csv(feature_file_path, delimiter='\t', header=None)
ordered_categories = features_df[1].unique().tolist()

# Apply color map
color = '#5ec962'
all_mea = []
R2_all = []
ch_names_ = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
             'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']
order_index = []

def extract_number(filename):
    match = re.search(r'prediction_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No se encontró un número en el nombre del archivo: {filename}")

# Read and process the files
for file in files:
    print(file)
    number = extract_number(file)
    if number == 18:
        ch_name = ch_names_[-1]
    else:
        ch_name = ch_names_[number]
    print(number)
    order_index.append(int(number))
    print(ch_name)

    # Read the file into a pandas DataFrame
    df = pd.read_csv(file, delimiter='\t', header=None)
    df.columns = ['Col1', 'Col2', 'Col3', 'Col4']

    # Aplicar la función exponencial a las columnas 'Col1' y 'Col2'
    df['Col1'] = df['Col1']#np.exp(df['Col1'])
    df['Col2'] = df['Col2']#np.exp(df['Col2'])

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={'aspect': 0.65})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mse = mean_squared_error(df['Col1'], df['Col2'])
    mae = mean_absolute_error(df['Col1'], df['Col2'])
    correlation = np.corrcoef(df['Col1'], df['Col2'])[0, 1]
    r2 = r2_score(df['Col1'], df['Col2'])

    # Add labels to scatter plot
    label = f"Channel {ch_name} All \nMAE: {mae:.2f}\n $R^2$: {r2:.2f}"
    ax.scatter(df['Col1'], df['Col2'], label=label, color=color, alpha=0.5)

    all_mea.append(mae)
    R2_all.append(r2)

    # Add y=x line
    min_val = min(df['Col1'].min(), df['Col2'].min())
    max_val = max(df['Col1'].max(), df['Col2'].max())
    ax.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='y=x')

    # Add labels, title, and legend
    #ax.set_ylim([-10, 101])
    #ax.axhline(0, color='grey', linestyle='dashed', linewidth=1)
    ax.set_xlim([0, 100])
    ax.set_xlabel('Age', fontsize=20)
    #ax.set_xlabel('Log(Age)', fontsize=20)
    ax.set_ylabel('Predicted age', fontsize=20)
    #ax.set_title('General Comparison', fontsize=20)
    #ax.set_ylabel('Log(Predicted age)', fontsize=20)
    ax.set_title('General Comparison', fontsize=20)
    ax.set_yticks(np.arange(0.0, 101, 20), fontsize=17)
    ax.set_xticks(np.arange(0.0, 101, 5), fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=18)
    plot_filename = os.path.join("plots", f'{os.path.basename(file)}_general_plot.png')
    plt.savefig(plot_filename)
    plt.close(fig)

    # Create violin plot
    fig, ax = plt.subplots(figsize=(7, 6))
    df['Col2_Col1_Diff'] = df['Col2'] - df['Col1']
    data = [df['Col2_Col1_Diff']]
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    ax.set_ylabel('Delta (Predicted - actual age)', fontsize=18)
    ax.tick_params(axis='y', labelsize=17)

    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    mean = np.mean(data)
    ax.plot(1, mean, 'wo', markersize=8, markeredgecolor='black')

    ax.set_xticks([1])
    ax.set_xticklabels(['All'], rotation=30, fontsize=17, ha="center")
    ax.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    fig.tight_layout()

    # Save plot
    violin_plot_filename = os.path.join("plots", f'{os.path.basename(file)}_general_violin_plot.png')
    plt.savefig(violin_plot_filename)
    plt.close()


# order indices
indices_ordenados = np.argsort(order_index)
order_index_ordered = np.array(order_index)[indices_ordenados]
all_mea_ordered = np.array(all_mea)[indices_ordenados]
R2_all_ordered = np.array(R2_all)[indices_ordenados]

print(order_index_ordered)
print(all_mea_ordered)

print("Scatter and violin plots have been created and saved for all files.")

print(all_mea)
# Bar plot
plt.figure(figsize=(10, 6))
for spine in plt.gca().spines.values():
        spine.set_visible(False)
plt.bar(np.arange(len(all_mea)), all_mea_ordered, color=color, label='All', alpha=0.5)
plt.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(8, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(10, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(12, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(14, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.xticks(np.arange(len(all_mea)), ch_names_, ha="right", fontsize=17)
plt.xlabel('Channel', fontsize=16)
plt.ylabel('MAE', fontsize=16)
plt.yticks(fontsize=17)
plt.title('Mean Absolute Error', fontsize=18)
#plt.legend(fontsize=14)
plt.savefig("plots/all_mea_bar_plot.png")
plt.close()

# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
for spine in plt.gca().spines.values():
        spine.set_visible(False)
plt.bar(np.arange(len(R2_all)), R2_all_ordered, color=color, label='All', alpha=0.5)
plt.xlabel('Channels', fontsize=16)
plt.ylabel('$R^2$', fontsize=16)
plt.xticks(np.arange(len(all_mea)), ch_names_, ha="right", fontsize=17)
plt.title('$R^2$ for All', fontsize=18)
plt.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.1, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.3, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
#plt.axhline(0.5, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
#plt.axhline(0.6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.yticks(fontsize=17)
plt.legend(fontsize=14)
plt.savefig("plots/all_r2_bar_plot.png")
plt.close()


pepe = list(R2_all_ordered)
variances = R2_all_ordered
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

pos_2d = np.array(pos_2d)

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

# we need to specify the sampling frequency, but it is not used and it doesn't matter what we put here
# add montage to info object - using the info object we do not need to project our 3D sensor locations
# to a 2D plane as mne will do this for us when plotting
info.set_montage(montage_true)

fig2, ax2 = plt.subplots(figsize=(10, 8))
img, _ = mne.viz.plot_topomap(evoked.data[:, 0], info, cmap='viridis', size=5, ch_type='eeg', sensors=True,
                              sphere=None, contours=4, outlines='head', names=ch_names, vlim=(vmin, vmax),
                              show=False, axes=ax2, mask=mask, mask_params=dic_)
cbar_max = 1
cbar_min = -0.2
cbar_step = 0.1

divider = make_axes_locatable(ax2)

cax = divider.append_axes("right", size="2%", pad=0.05)
cbar = plt.colorbar(ax=ax2, shrink=0.7, orientation='vertical', mappable=img,
                    ticks=[0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], cax=cax)
cbar.set_label(r'$R^2$', size=20)

for t in cbar.ax.get_yticklabels():
    t.set_fontsize(20)
cbar.set_label(r'KMER $R^2$',size=20)

figname = "plots/k_eeg_head.png"
fig2.tight_layout()
plt.savefig(figname,dpi=200)
plt.close()
print("Bar plots have been created and saved.")
