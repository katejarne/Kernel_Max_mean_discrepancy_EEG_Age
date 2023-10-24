######################################################################################
#               C.Jarne- 2023 Analysis Group D. Vidahurre  @cfin                     #
# This Python code loads .mat files from a directory of the data set HarMNqEEG,      #
# reads data from those files (https://doi.org/10.1016/j.neuroimage.2022.119190)     #
# and normalizes it. The normalized data is then stored in a dictionary with each    #
# key containing a specific channel’s normalized spectrum to create a  distance      #
# estimation matrix. After that, it creates a distance matrix for each kernel using  #
# the mmd_def and mmd_estimation_vec functions. Finally, it prints out some          #
# information on the loaded data such as the number of subjects, the number of       #
# frequency ranges, and the number of channels.                                      #
# input:                                                                             #
# .mat files in directory                                                            #
# output:                                                                            #
# feature file and mmd distance matrice of combined channels per each kernel         #
######################################################################################

import os
import scipy.io as sio  # for matlab file reading
# Defined in the directory
from mmd_estimation_vec import *

start_time = time.time()
feature_list = []
subjet_data_ch = []

# Directory with Raw matlab files from the datase HarMNqEEG
# To test code use a short selection otherwise it takes time

current_directory = os.path.dirname(__file__)
root_dir = current_directory + "/sub_selection_test/"
# Dictionary for each channel list for data
channel_dic_data = {'subjet_list_hist_data_ch_{}'.format(i):[] for i in np.arange(19)}
id_names = []

def get_files_in_subdirectories(root_dir):
    file_names = []
    subdir_names = []
    ages = []
    sexes = []
    freqranges = []
    fmaxes = []
    fmins = []
    freqress = []
    Spectrums = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mat"):
                file_path = os.path.join(subdir, file)
                data = sio.loadmat(file_path)
                file_names.append(file)
                subdir_names.append(os.path.basename(subdir))
                ages.append(float(data['data_struct'][0][0]['age'][0][0]))
                gender = str(data['data_struct'][0][0]['sex'])
                sexes.append(gender)
                range_f = data['data_struct'][0]['freqrange'][0][0]
                freqranges.append(len(range_f))
                fmaxes.append(data['data_struct'][0][0]['fmax'][0][0])#
                fmins.append(data['data_struct'][0][0]['fmin'][0][0])#
                freqress.append(data['data_struct'][0][0]['freqres'][0][0])#
                full_spect = data['data_struct'][0][0]['Spec'][0:19]
                
                # All channels histograms are saved in .txt if needed (here they are not normalized- Be aware)
                # aca = np.c_[np.array(full_spect[:, 0:49]).T]
                # 49 is to unify size of histograms to consider same freq. range
                # name_file = os.path.basename(file) + ".txt"
                # np.savetxt(name_file, aca, fmt='%d\t' * 19, delimiter='\n')
                
                id_names.append(os.path.basename(subdir) + "_" + os.path.basename(file))
                aux = []
                vector1 = np.arange(17)
                vector = np.append(vector1, 18)
                
                # To create an array that contains all channel histograms
                # Channels are ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
                # 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
                # Cz excluded for been reference (ground).

                for ch in vector:
                    # print("ch",ch)
                    channel_dic_data["subjet_list_hist_data_ch_" +
                                     str(ch)].append(full_spect[ch][0:49]/np.sum(full_spect[ch][0:49]))
                    # Each channel in the dictionary has a normalized spectrum to distance estimation matrix
                    # (otherwise distance matrices are wrong)
                    aux.append(full_spect[ch][0:49]/np.sum(full_spect[ch][0:49]))
                aux = np.array(aux).reshape(len(aux), -1)
                subjet_data_ch.append(aux)
                Spectrums.append(full_spect)
                # print("Frequency range", len(range_f))

    return file_names, subdir_names, sexes, ages, fmaxes, fmins, freqress, freqranges
#print("--- %s seconds to loop---" % (time.time() - start_time))

def write_to_file(file_names, subdir_names, sexes, ages,fmaxes, fmins, freqress, freqranges, output_file):
    with open(output_file, "w") as f:
        for file, subdir, sex, age, fmax, fmin, freqres, freqrange\
                in zip(file_names, subdir_names, sexes, ages, fmaxes, fmins, freqress, freqranges):
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(file, subdir, sex,
                                                              float(age), fmax, fmin, freqres,freqrange))

file_names, subdir_names, ages, sexes, fmaxes, fmins, freqress,freqranges = get_files_in_subdirectories(root_dir)
write_to_file(file_names, subdir_names, ages, sexes, fmaxes, fmins, freqress,freqranges, "out/salida_features.txt")

feature_list = np.c_[subdir_names, ages, ages]

# print("Subjects number: ",len(subjet_data_ch))
# print("Subject numer of frequencies: ", len(subjet_data_ch[0]))
# print("subjet number of channels: ", len(subjet_data_ch[0][1]))

# To create a distance matrix for each kernel
     
list_matrix_linear = []
list_matrix_pol = []
list_matrix_cos = []
list_matrix_rbf = []
          
for j in range(len(subjet_data_ch)):
    print("id_names", id_names[j])
    # calling mmd_estimation estimation function
    pepe = mmd_estimation(subjet_data_ch, subjet_data_ch[j])

    list_matrix_pol.append(pepe[0])
    list_matrix_cos.append(pepe[1])
    list_matrix_rbf.append(pepe[2])
    list_matrix_linear.append(pepe[3])

    distance_matrix_pol = list_matrix_pol
    distance_matrix_cos = list_matrix_cos
    distance_matrix_rbf = list_matrix_rbf
    distance_matrix_lin = list_matrix_linear
    
# Distance matrices for 4 different Kernel
np.savetxt('out_combined/eeg_mmd_matrix_c_pol.txt',distance_matrix_pol,delimiter='\t')
# np.savetxt('out_combined/eeg_mmd_matrix_c_cos.txt',distance_matrix_cos,delimiter='\t')
np.savetxt('out_combined/eeg_mmd_matrix_c_rbf.txt',distance_matrix_rbf,delimiter='\t')
np.savetxt('out_combined/eeg_mmd_matrix_c_lin.txt',distance_matrix_lin,delimiter='\t')
     
print("All matrices created in folder: /out_combined")         
