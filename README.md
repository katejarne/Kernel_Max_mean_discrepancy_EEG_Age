
# Kernel Mean Embbeading Regression (KMER) with EEG frequency spectrum used for Age prediction:

This is the framework used for the paper "Predicting subject traits from brain spectral signatures: a case for brain ageing"

This repository contains code to perform age prediction applying BR, KRR and KMER using EEG from the HarMNqEEG (https://doi.org/10.1016/j.neuroimage.2022.119190) data set based on the frequency content and using  Maximum Mean Discrepancy with different kernels.

Multinational EEG cross-spectrum and anonymized metadata come from 9 countries, 12 devices, and 14 sites, including 1966 subjects, and is hosted in \url{https://www.synapse.org/} with id: $syn26712693$. Complete access is possible through registration and login to the system.

The framework consists of 3 parts: one to estimate distance matrices in the file `Estimation_of_distance_matrices`, the other to generate the plots from the paper `Generate_plots` and one part to make the predictions using  three methods (`main`).

## Dependencies
Numpy 
Scipy
Sckit learn
MNE-Python (for data visualization)

## Files included

----------------------------------------------
I) To generate Maximum Mean Discrepancy Distance matrices, use the code in the folder `Estimation_of_distance_matrices`:

1. Use `main_eeg.py` to generate individual matrices per channel and type of kernel (it will generate all matrices for every channel and the four kernels).
2. Use `main_eeg_for_combined.py` to generate a Distance matrix combining all channels (Not implemented).

For both cases, the HarMNqEEG data set is needed. Download it from: [https://10.0.28.135/syn26712693](https://10.0.28.135/syn26712693) as indicated in the original paper: [https://www.synapse.org/#!Synapse:syn26712693/files/](https://www.synapse.org/#!Synapse:syn26712693/files/).

The code is commented in detail.

The following functions are called:
- `mmd_def.py`: Definition of MMD empirical Distance for each kernel.
- `mmd_estimation_ind.py`: Estimation of distance for individual channel data.
- `mmd_estimation_vec.py`: Estimation of distance for combined channel data.

II) To estimate age using MMD matrices, run the following code with the feature file `features_ok.txt`, which is located in `input_files/Feature_file`, and the path with Distance matrices as input:

`KMER_ind_channels.py`

III) To estimate age using histograms as features, run the following code with the feature file `features_ok.txt`, which is located in `input_files/Feature_file`, and histograms available and compressed in `individual_raw_spectrums.zip`. They have been generated using the `main_eeg.py` function.

`BR_ind_channels.py`
`KRR_ind_channels.py` for a kernelized version

Note: II) and III) implement a leave-one-site-out cross-validation strategy to estimate the best model

IV) To plot the results, use the following files in the folder Generate_plots with its readme.txt document

Paper results are available in the `results-predictions` folder in `Generate_plots`. Distance matrices have to be built with the code, or [here](https://www.dropbox.com/scl/fi/x30yptc9vro7501g3etek/kernel_gauss.zip?rlkey=ozdi0suezldi95reek5nunsxz&dl=0)


