
# Kernel Mean Embbeading Regression (KMER) or Kernel Ridge Regression with EEG frequency spectrum used for Age prediction:

This is the framework used for the paper "Predicting subject traits from brain spectral signatures: a case for brain ageing"

This repository contains code to perform age prediction applying BR, KRR and KMER using EEG from the HarMNqEEG (https://doi.org/10.1016/j.neuroimage.2022.119190) data set based on the frequency content and using  Maximum Mean Discrepancy with different kernels.

Multinational EEG cross-spectrum and anonymized metadata come from 9 countries, 12 devices, and 14 sites, including 1966 subjects, and is hosted in \url{https://www.synapse.org/} with id: $syn26712693$. Complete access is possible through registration and login to the system.

The framework consists of 2 scripts one to perform a Ridge regression (`BR_ind_channels.py`) and the other to perform KMER or KRR (`KMER-KRR_ind_channels.py`) depending on method selection. We also include a third script `KRR_ind_channels_bias_corrected.py`, which performs also KMER and KRR regression and includes a bias correction.
## Dependencies
Sure! Here are the software names in markdown format with links:

- [Numpy](https://numpy.org/)
- [Scipy](https://www.scipy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [MNE-Python (for data visualization)](https://mne.tools/stable/index.html)
## Files included


For both cases, the HarMNqEEG data set is needed. Download it from: [https://10.0.28.135/syn26712693](https://10.0.28.135/syn26712693) as indicated in the original paper: [https://www.synapse.org/#!Synapse:syn26712693/files/](https://www.synapse.org/#!Synapse:syn26712693/files/). Or use the .txt in invididual_raw_spectums.zip that includes 19 columns for each channel.

The code is commented in detail.

The following functions are called:
- `mmd_definition.py`: Definition of MMD empirical Distance for each kernel.

To estimate age run the following code with the feature file `features_ok.txt`, which is located in `input_files/Feature_file`.

Note: a cross-validation strategy to estimate the best model in all cases. Code respects group structure for CV.

IV) To plot the results, use the following files in the folder Generate_plots with its readme.txt document

Paper results are available in the `Results_files.zip` folder in `Generate_plots`. 

