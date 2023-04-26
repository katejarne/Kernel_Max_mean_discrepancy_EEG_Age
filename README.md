# Kernel Maximum Mean Discrepancy EEG used for Age prediction:

This repository contains code to perform age prediction using EEG from the HarMNqEEG (https://doi.org/10.1016/j.neuroimage.2022.119190) data set and Maximum Mean Discrepancy with different kernels.

## Files included

I) To generate Maximum Mean Discrepancy Distance matrices, use the code in the folder `Estimation_of_distance_matrices`:

1. Use `main_eeg.py` to generate individual matrices per channel and type of kernel (it will generate all matrices for every channel and the four kernels).
2. Use `main_eeg_for_combined.py` to generate a Distance matrix combining all channels.

For both cases, the HarMNqEEG data set is needed. Download it from: [https://10.0.28.135/syn26712693](https://10.0.28.135/syn26712693) as indicated in the original paper: [https://www.synapse.org/#!Synapse:syn26712693/files/](https://www.synapse.org/#!Synapse:syn26712693/files/).

Code is commented in detail.

The following functions are called:
- `mmd_def.py`: Definition of MMD empirical Distance for each kernel.
- `mmd_estimation_ind.py`: Estimation of distance for individual channel data.
- `mmd_estimation_vec.py`: Estimation of distance for combined channel data.

II) To estimate age using MMD matrices, run the following code with the feature file `salida_features_ok.txt`, which is located in `input_files/Feature_file`, and the path with Distance matrices as input:

`regression_kernel_ridge.py`

III) To estimate age using histograms as features, run the following code with the feature file `salida_features_ok.txt`, which is located in `input_files/Feature_file`, and histograms available and compressed in `individual_raw_spectrums.zip`. They have been generated using the `main_eeg.py` function.

`regression_ridge.py`

Note: II) and III) implements a fold cross validation strategy to estimate the best model

IV) To plot the results, use the following files in folder Generate_plots:
1. `plot_correlation_EEG_age_pred_real.py`: For correlation between predicted and real age
2. `plot_correlation_points_color_groups_age.py`: For correlation between predicted and real age separating in sites
3. `plot_eeg_head_example.py`: Plot variance in each channel
4. `plot_mmd_distance_matrices.py`: Plot distance matrices

"Paper results are available in the `output_files` folder. Distance matrices have to be built with the code, or requested, because their size does not allow us to save them in the GitHub repository."

