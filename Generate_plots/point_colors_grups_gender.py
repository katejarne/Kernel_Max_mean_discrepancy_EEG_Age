import os
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

# Directory where the files are located
directory = "Results_files/KMER_gamma_free_0.1-1-10/"
# Get all files matching the pattern
files = glob.glob(os.path.join(directory, "KMER_prediction_*.txt"))
#files = glob.glob(os.path.join(directory, "KRR_prediction_*.txt"))

# Leer el archivo de features y extraer las categorías
feature_file_path = "..Feature_file/features_ok.txt"
features_df = pd.read_csv(feature_file_path, delimiter='\t', header=None)
ordered_categories = features_df[1].unique().tolist()

# Categories to be used
categories = ["['F']","['M']"]
category_labels = {"['F']": 'female',"['M']": 'male'}
# Apply color map
colors = plt.cm.viridis(np.linspace(0, 1, 3))
male_mea = []
female_mea = []
R2_male =[]
R2_female=[]
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
        ch_name=ch_names_[-1]
    else:
        ch_name=ch_names_[number]
    print(number)
    order_index.append(int(number))
    print(ch_name)

    # Read the file into a pandas DataFrame
    df = pd.read_csv(file, delimiter='\t', header=None)
    df.columns = ['Col1', 'Col2', 'Col3', 'Col4']


    df['Col1'] = df['Col1'] # np.exp(df['Col1'])
    df['Col2'] = df['Col2'] # np.exp(df['Col2'])

    # Filter data for the categories of interest
    df = df[df['Col3'].isin(categories)]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7)) #, subplot_kw={'aspect': 0.65})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, category in enumerate(categories):
        subset = df[df['Col3'] == category]
        mse = mean_squared_error(subset['Col1'], subset['Col2'])
        mae = mean_absolute_error(subset['Col1'], subset['Col2'])
        correlation = np.corrcoef(subset['Col1'], subset['Col2'])[0, 1]
        r2 = r2_score(subset['Col1'], subset['Col2'])

        # Add labels to scatter plot
        label = f"Channel {ch_name} {category_labels[category]} \nMAE: {mae:.2f}\n $R^2$: {r2:.2f}"
        # ax.scatter(subset['Col1'], subset['Col2'], label=category_labels[category] +label, color=colors[i], alpha=0.5)
        # Add y=x line
        min_val = min(df['Col1'].min(), df['Col2'].min())
        max_val = max(df['Col1'].max(), df['Col2'].max())
        # ax.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='y=x')

        if i == 1:  # male
            male_mea.append(mae)
            R2_male.append(r2)
            fig_male, ax_male = plt.subplots(figsize=(10, 7))#, subplot_kw={'aspect': 0.65})
            ax_male.spines['top'].set_visible(False)
            ax_male.spines['right'].set_visible(False)
            ax_male.scatter(subset['Col1'], subset['Col2'], label=label, color=colors[i], alpha=0.5)
            ax.scatter(subset['Col1'], subset['Col2'], label=label, color=colors[i], alpha=0.5)
            ax_male.set_ylim([-10, 101])
            ax_male.axhline(0, color='grey', linestyle='dashed', linewidth=1)
            ax_male.set_xlim([0, 100])
            ax_male.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='y=x')
            ax_male.set_xlabel('Age', fontsize=20)
            ax_male.set_ylabel('Predicted age', fontsize=20)
            ax_male.set_yticks(np.arange(0.0, 101, 20), fontsize=17)
            ax_male.set_xticks(np.arange(0.0, 101, 10), fontsize=17)
            ax_male.tick_params(axis='both', which='major', labelsize=17)
            ax_male.legend(loc=2, fontsize=18) #, bbox_to_anchor=(1.0, 0.5)
            plot_filename = os.path.join("plots", f'{os.path.basename(file)}_m_gender_plot.png')
            fig_male.tight_layout()
            fig_male.savefig(plot_filename)
            plt.close(fig_male)
        if i == 0:  # female
            female_mea.append(mae)
            R2_female.append(r2)
            fig_female, ax_female = plt.subplots(figsize=(10, 7))#, subplot_kw={'aspect': 0.65})
            ax_female.spines['top'].set_visible(False)
            ax_female.spines['right'].set_visible(False)
            ax_female.scatter(subset['Col1'], subset['Col2'], label=label, color=colors[i], alpha=0.5)
            ax.scatter(subset['Col1'], subset['Col2'], label=label, color=colors[i], alpha=0.5)
            ax_female.set_ylim([-10, 101])
            ax_female.axhline(0, color='grey', linestyle='dashed', linewidth=1)
            ax_female.set_xlim([0, 100])
            ax_female.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='y=x')
            ax_female.set_xlabel('Age', fontsize=20)
            ax_female.set_ylabel('Predicted age', fontsize=20)
            ax_female.set_yticks(np.arange(0.0, 101, 20), fontsize=17)
            ax_female.set_xticks(np.arange(0.0, 101, 10), fontsize=17)
            # ax_female.tick_params(axis='both', which='major', labelsize=20)
            ax_female.tick_params(axis='both', which='major', labelsize=17)
            ax_female.legend(loc=2, fontsize=18) #, bbox_to_anchor=(1.0, 0.5)
            fig_female.tight_layout()
            plot_filename = os.path.join("plots", f'{os.path.basename(file)}_f_gender_plot.png')
            fig_female.savefig(plot_filename)
            plt.close(fig_female)

    # Add labels, title, and legend
    ax.set_ylim([-10, 101])
    ax.axhline(0, color='grey', linestyle='dashed', linewidth=1)
    ax.set_xlim([0, 100])
    ax.set_xlabel('Age', fontsize=20)
    ax.set_ylabel('Predicted age', fontsize=20)
    ax.set_title('Gender comparison', fontsize=20)
    ax.set_yticks(np.arange(0.0, 101, 20), fontsize=17)
    ax.set_xticks(np.arange(0.0, 101, 5), fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=18)
    plot_filename = os.path.join("plots", f'{os.path.basename(file)}_gender_plot.png')
    plt.savefig(plot_filename)
    plt.close(fig)

    # Create violin plot for each category
    fig, ax = plt.subplots(figsize=(7, 6))
    data = df['Col2_Col1_Diff'] = df['Col2'] - df['Col1']
    data = [df[df['Col3'] == category]['Col2_Col1_Diff'] for category in categories]
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    ax.set_ylabel('Delta (Predicted - actual age)', fontsize=18)
    ax.tick_params(axis='y', labelsize=17)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    means = [np.mean(d) for d in data]
    for i, mean in enumerate(means):
        ax.plot(i + 1, mean, 'wo', markersize=8, markeredgecolor='black')

    ax.set_xticks(np.arange(1, len(categories) + 1))
    ax.set_xticklabels([category_labels[cat] for cat in categories], rotation=30, fontsize=17, ha="center")
    ax.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
    fig.tight_layout()

    # Save plot
    violin_plot_filename = os.path.join("plots", f'{os.path.basename(file)}_gender_violin_plot.png')
    plt.savefig(violin_plot_filename)
    plt.close()

indices_ordenados = np.argsort(order_index)
order_index_ordered = np.array(order_index)[indices_ordenados]
male_mea_ordered = np.array(male_mea)[indices_ordenados]
female_mea_ordered = np.array(female_mea)[indices_ordenados]
R2_male_ordered = np.array(R2_male)[indices_ordenados]
R2_female_ordered = np.array(R2_female)[indices_ordenados]

print("Scatter and violin plots have been created and saved for all files.")

plt.figure(figsize=(10, 6))
for spine in plt.gca().spines.values():
        spine.set_visible(False)
# plt.bar(np.arange(len(male_mea)), male_mea, color=colors[0], label='Male', alpha=0.5)
plt.bar(np.arange(len(male_mea)),male_mea_ordered, color=colors[1], label='Male', alpha=0.5)
plt.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(8, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(10, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(12, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.xticks(np.arange(len(female_mea)), ch_names_, ha="right", fontsize=17)
plt.xlabel('Channels', fontsize=16)
plt.ylabel('MAE', fontsize=16)
plt.yticks(fontsize=17)
#plt.title('Mean Absolute Error for Male', fontsize=18)
plt.legend(fontsize=14)
plt.savefig("plots/male_mea_bar_plot.png")
plt.close(fig)

plt.figure(figsize=(10, 6))
for spine in plt.gca().spines.values():
        spine.set_visible(False)
# plt.bar(np.arange(len(female_mea)), female_mea, color=colors[1], label='Female', alpha=0.5)
plt.bar(np.arange(len(female_mea)), female_mea_ordered, color=colors[0], label='Female', alpha=0.5)
plt.xlabel('Channels', fontsize=16)
plt.ylabel('MAE', fontsize=16)
plt.xticks(np.arange(len(female_mea)), ch_names_, ha="right", fontsize=17)
plt.axhline(2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(8, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(10, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(12, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
# plt.title('Mean Absolute Error for Female', fontsize=18)
plt.yticks(fontsize=17)
plt.legend(fontsize=14)
plt.savefig("plots/female_mea_bar_plot.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
for spine in plt.gca().spines.values():
    spine.set_visible(False)
# plt.bar(np.arange(len(R2_male)), R2_male, color=colors[0], label='Male', alpha=0.5)
plt.bar(np.arange(len(R2_male)), R2_male_ordered, color=colors[1], label='Male', alpha=0.5)
plt.xlabel('Channels', fontsize=16)
plt.ylabel('$R^2$', fontsize=16)
plt.xticks(np.arange(len(female_mea)), ch_names_, ha="right", fontsize=17)
plt.title('$R^2$ for Male', fontsize=18)
plt.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.1, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.3, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.5, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.7, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.8, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.yticks(np.arange(0.0, 0.85, 0.1), fontsize=15)
plt.legend(fontsize=14)
plt.savefig("plots/R2_male_bar_plot.png")
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6))
for spine in plt.gca().spines.values():
    spine.set_visible(False)
#plt.bar(np.arange(len(R2_female)), R2_female, color=colors[1], label='Female', alpha=0.5)
plt.bar(np.arange(len(R2_female)), R2_female_ordered, color=colors[0], label='Female', alpha=0.5)
plt.xticks(np.arange(len(female_mea)), ch_names_, ha="right", fontsize=17)
plt.xlabel('Channels', fontsize=16)
plt.axhline(0, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.2, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.1, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.3, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.4, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.5, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.6, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.7, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.axhline(0.8, color='grey', linestyle='dashed', linewidth=1, alpha=0.5)
plt.ylabel('$R^2$', fontsize=16)
plt.title('$R^2$ for Female', fontsize=18)
plt.yticks(np.arange(0.0, 0.85, 0.1), fontsize=15)
plt.legend(fontsize=14)
plt.savefig("plots/R2_female_bar_plot.png")
plt.close(fig)

if not (len(R2_male) == len(R2_female) == len(ch_names_)):
    raise ValueError("Las listas R2_male, R2_female y ch_names_ deben tener la misma longitud.")
print("ACA",len(R2_male), len(R2_female), print(ch_names_))
y = np.arange(0.7, 0.8, 0.1)
#y = np.arange(0.15,0.55,0.1)
plt.figure(figsize=(10, 6))
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(y, y, color='gray', linestyle='--', label='y=x',alpha=0.5)

for i, nombre in enumerate(ch_names_):
    plt.scatter(R2_male_ordered[i], R2_female_ordered[i], label=nombre)

plt.xlabel(r'$R^2$ male', fontsize=18)
plt.ylabel(r'$R^2$ female', fontsize=18)
plt.title(r'$R^2$ per channel male vs female', fontsize=18)
#plt.xlim([0.2,0.5])
#plt.xticks(np.arange(0.2,0.51,0.05), fontsize=18)
#plt.yticks(np.arange(0.2,0.51,0.05), fontsize=18)
#plt.ylim([0.2,0.5])

plt.legend()
plt.savefig("plots/gender_r2_compa.png")
plt.close(fig)


def head_plot(variances,gender):
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
    vmin, vmax = (np.min(evoked.data)), (np.max(evoked.data))#0.15,0.45#
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
    # axh.set_title("Accuracy MMD Kernel "+str(nombre_archivo), fontsize=25)
    # fig2.suptitle("Accuracy KMER "+str(nombre_archivo[-3:])+" Kernel", fontsize=20)

    img, _ = mne.viz.plot_topomap(evoked.data[:, 0], info, cmap='viridis', size=5, ch_type='eeg', sensors=True,
                                  sphere=None, contours=4, outlines='head', names=ch_names, vlim=(vmin, vmax),
                                  show=False, axes=ax2, mask=mask, mask_params=dic_)

    cbar_max = 1
    cbar_min = 0
    cbar_step = 0.1

    divider = make_axes_locatable(ax2)

    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(ax=ax2, shrink=0.7, orientation='vertical', mappable=img,
                        ticks=[0, 0.1, 0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], cax=cax)
    cbar.set_label(r'$R^2$', size=20)

    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(20)
    cbar.set_label(r'KMER $R^2$',size=20)

    figname = f"plots/k_eeg_head{gender}.png"
    fig2.tight_layout()
    plt.savefig(figname,dpi=200)
    plt.close()

pepe = head_plot(R2_female_ordered,"female")
pepe = head_plot(R2_male_ordered,"male")

diference = np.array(R2_female) - np.array(R2_male)

res = stats.wilcoxon(diference, alternative='greater')
print("Difference",diference)
print("Test result ", res)
print("---------------")

# Generate output table
with open(f"full_KMER_r2_male.txt", "w") as f:
    f.write("k\tR2 per channel\n")
    for k, res in enumerate(R2_male):
        f.write(f"{k}\t{res}\n")

# Generate output table
with open(f"full_KMER_r2_female.txt", "w") as f:
    f.write("k\tR2 per channel\n")
    for k, res in enumerate(R2_female):
        f.write(f"{k}\t{res}\n")
