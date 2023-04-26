# Kernel Maximum Mean Discrepancy EEG used for Age prediction:

This repository contains code to perform age prediction using EEG from the HarMNqEEG (https://doi.org/10.1016/j.neuroimage.2022.119190) data set and Maximum Mean Discrepancy with different kernels.

## Files included
To plot the results, use the following files in folder Generate_plots:
1. `plot_correlation_EEG_age_pred_real.py`: For correlation between predicted and real age
2. `plot_correlation_points_color_groups_age.py`: For correlation between predicted and real age separating in sites
3. `plot_eeg_head_example.py`: Plot variance in each channel
4. `plot_mmd_distance_matrices.py`: Plot distance matrices

To generate Maximum Mean Discrepancy Distance matrices, use the code in the folder `Estimation_of_distance_matrices`:

1. Use `main_eeg.py` to generate individual matrices per channel and type of kernel (it will generate all matrices for every channel and the four kernels).
2. Use `main_eeg_for_combined.py` to generate a Distance matrix combining all channels.

The following files are called:
- `mmd_def.py`: Definition of MMD empirical Distance for each kernel.
- `mmd_estimation_ind.py`: Estimation of distance for individual channel data.
- `mmd_estimation_vec.py`: Estimation of distance for combined channel data.
