##################################################################################################################
#                            C.Jarne- 2023 Analysis Group D. Vidahurre  @cfin                                    #
#                                                                                                                #
# The code takes a directory, loops through all the files and loads data from .mat files of                      #
# HarMNqEEG data set (https://doi.org/10.1016/j.neuroimage.2022.119190)                                          #
# Defines a dictionary "channel_dic_data" to store the data for each channel and initializes. Data is then       #
# processed and stored in lists and dictionaries ("subjet_data_ch," "id_names," and "channel_dic_data.")         #
# This function also saves the processed data to a file. Code calls a function to write the data to a text file. #                                  #
# Code uses a for loop to iterate over values of ch in the range. Inside the loop, the code retrieves data       #
# from a dictionary called channel_dic_data using the key "subjet_list_hist_data_ch_"+str(ch).                   #
# Data is then stored in a list called by iterating over the length of the subjet_data_ch list                   #
# and appending the corresponding data to the list_data list. Next, the code initializes 4 empty lists.          #
# These lists will store distance matrices for each kernel and ch. Then, with another loop iterates over         #
# the length of the subjet_data_ch list. Inside this loop, the code prints the ID name of the current iteration, #
# then calls mmd_estimation function. This function returns four values (Maximum Mean Discrepancy) Distances,    #
# appended to the respective distance matrix lists. Finally, the code prints the list_matrix_linear list.        #
#                                                                                                                #
# input:                                                                                                         #
# .mat files in directory                                                                                        #
# output:                                                                                                        #
# feature file and mmd distance matrices per channel                                                             #
##################################################################################################################

import os
import scipy.io as sio   # for matlab file reading
from mmd_estimation_ind import *

# Directory with Raw matlab files from the datase HarMNqEEG
# To test code use a short selection otherwise it takes time

current_directory = os.path.dirname(__file__)
root_dir = current_directory + "/sub_selection_test/"

start_time = time.time()
feature_list = []
subjet_data_ch = []
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
    fmins  = []
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

                # All channels histograms are saved in .txt (here they are not normalized. Be aware)
                aca = np.c_[np.array(full_spect[:, 0:49]).T]
                # 49 is to unify size of histograms to consider same freq. range
                name_file = os.path.basename(file) + ".txt"
                np.savetxt("histos/"+name_file, aca, fmt='%d\t' * 19, delimiter='\n')
                
                subjet_data_ch.append(aca)
                
                id_names.append(os.path.basename(subdir) + "_" + os.path.basename(file))
                for ch in np.arange(19):
                    print("ch", ch)
                    # Each channel in the dictionary has a normalized spectrum
                    # to distance estimation matrix (otherwise distance matrices are wrong)
                    channel_dic_data["subjet_list_hist_data_ch_"
                                     + str(ch)].append(full_spect[ch][0:49]/np.sum(full_spect[ch][0:49]))
                Spectrums.append(full_spect)
                print("Frequency range", len(range_f))

    return file_names, subdir_names, sexes, ages, fmaxes, fmins, freqress, freqranges
# print("--- %s seconds to loop---" % (time.time() - start_time))

def write_to_file(file_names, subdir_names, sexes, ages,fmaxes, fmins, freqress, freqranges, output_file):

    with open(output_file, "w") as f:
        for file, subdir, sex, age, fmax, fmin, freqres, freqrange \
                in zip(file_names, subdir_names,sexes, ages, fmaxes, fmins, freqress,freqranges):
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(file, subdir, sex,
                                                              float(age), fmax, fmin, freqres,freqrange))

file_names, subdir_names, ages, sexes, fmaxes, fmins, freqress,freqranges = get_files_in_subdirectories(root_dir)
write_to_file(file_names, subdir_names, ages, sexes, fmaxes, fmins, freqress,freqranges, "out/features_ok.txt")
print("Feature file created in folder /out")


feature_list = np.c_[subdir_names, ages, ages]

# print("Subjects number: ", len(subjet_data_ch))
# print("Subject number of frequencies: ", len(subjet_data_ch[0]))
# print("Subjet number of channels: ", len(subjet_data_ch[0][1]))

# Comment following lines if you only want the histograms and feature files and not estimate distance matrices
# Channels are
# ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
# Cz excluded for been reference (ground).

channels = list(np.arange(0, 17, 1))
channels.append(18)

for ch in channels:
    data_subjet_1 = channel_dic_data["subjet_list_hist_data_ch_"+str(ch)]
    list_data = []
    for i in range(len(subjet_data_ch)):
        list_data.append(data_subjet_1[i])

    # Create a distance matrix for each considered kernel
     
    list_matrix_linear = []
    list_matrix_pol = []
    list_matrix_cos = []
    list_matrix_rbf = []
          
    for j in range(len(subjet_data_ch)):
        print("id_names", id_names[j])
        
        # calling mmd_estimation estimation function
        pepe = mmd_estimation(list_data, data_subjet_1[j])
           
        print("Distance estimation", pepe)
        list_matrix_pol.append(pepe[0])
        list_matrix_cos.append(pepe[1])
        list_matrix_rbf.append(pepe[2])
        list_matrix_linear.append(pepe[3])

        distance_matrix_pol = list_matrix_pol
        distance_matrix_cos = list_matrix_cos
        distance_matrix_rbf = list_matrix_rbf
        distance_matrix_lin = list_matrix_linear
        
    # Distance matrices for 4 differente Kernel
    np.savetxt('out/eeg_mmd_matrix'+str(ch)+'_pol.txt', distance_matrix_pol,delimiter='\t')
    #    np.savetxt('out/eeg_mmd_matrix'+str(ch)+'_cos.txt', distance_matrix_cos,delimiter='\t')
    np.savetxt('out/eeg_mmd_matrix'+str(ch)+'_rbf.txt', distance_matrix_rbf,delimiter='\t')
    np.savetxt('out/eeg_mmd_matrix'+str(ch)+'_lin.txt', distance_matrix_lin,delimiter='\t')

print("All matrices created in folder /out")               
