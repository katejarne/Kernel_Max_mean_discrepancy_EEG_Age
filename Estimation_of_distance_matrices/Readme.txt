You need to create the following directories to run the code to build the distance matrices:

1) out (here distance matrices will be created)
2) out_combined (combined distance matrices will be created)
3) histos (output histograms will be created

input directory should be your .mat files from the dataset (raw data)

Run main_eeg.py to generate the matrices per channel (and the Feature file and individual histograms)
Run main_eeg_for_combined.py to generate the combined distance matrices (not implemented in the paper)

The definition of kernel distances is in mmd_def.py
The function to actually calculate the distance is mm_estimation_ind.py
