####################################################################################################
#                   C. Jarne 2023 Analysis Group of D. Vidahurre  @cfin                            #
#                                                                                                  #
# This  Python code performs kernel ridge regression for EEG Distance matrices in directory (KMER) #
# The script starts by defining a list of folds index (which correspond to each site),             #
# then it reads the files from a directory, loads the target values and masks the data based       #
# on age or gender, then iterates over the files in the directory. For each file, it loads the     #
# MMD distance matrix and applies the mask, then iterates over the folds, splits the data          #
# into training and testing sets, trains a kernel ridge regression model on the training data,     #
# predicts the target values for the testing data, and saves the model and the predicted values.   #
# Finally, it saves the results to a text file of the R-squared score, and calculates              #
# Pearson/Sperman correlation coefficient between the predicted and actual values.                 #
#                                                                                                  #
# input:                                                                                           #
# Distance matrices in dir                                                                         #
# output:                                                                                          #
# KMER_r2_+...+.txt with accuracy                                                                  #
# KMER_prediction+...+.txt with real vs predicted age                                              #
####################################################################################################

import os
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

# Folds defined by site (batches)
folds_struct = [
    list(range(0, 40)),       # Chendu
    list(range(40, 86)),      # Bern
    list(range(86, 359)),     # Chongqing
    list(range(359, 492)),    # CHBMP
    list(range(492, 791)),    # Russia
    list(range(791, 989)),    # Cuba90
    list(range(989, 1003)),   # Cuba2004
    list(range(1003, 1206)),  # Germany
    list(range(1206, 1261)),  # Cuba2003
    list(range(1261, 1331)),  # Barbados
    list(range(1331, 1359)),  # Malaysia
    list(range(1359, 1380)),  # Colombia
    list(range(1380, 1756)),  # NewYork
    list(range(1756, 1966))   # Switzerland
]

# Get the list of files in the directory

current_directory = os.path.dirname(__file__)
directory = current_directory + '/input_files/EEG_normalized_distance_matrices/'
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Load target values (Feature file)
data_ = np.loadtxt(current_directory+'/input_files/Feature_file/features_ok.txt', delimiter='\t', usecols=(3))
data = np.nan_to_num(data_)

gender = np.loadtxt(current_directory+'/input_files/Feature_file/features_ok.txt', delimiter='\t', usecols=(2), dtype=str)
print("gender", gender)

# Bolean mask to select age groups, gender or other features
mask = (data >= 5)  # for gender sel (gender == "['F']")

# Select only the values of "data" and "X" that meet the mask condition
data_masked = data[mask]

# index that we used later for selection
indices_mask = np.where(mask)[0]
print("index len i. e. number of individuals that match the criteria: ", len(indices_mask))

# Number of bootstrapping iterations (For uncertanty estimation)
n_bootstrap = 5

folds_struct_filtered = []

# alpha values
alpha_values = [100, 50, 0.3, 0.04, 2, 10, 1, 0.5, 0.001, 0.03, 0.1, 0.05, 0.2, 0.7, 0.01, 0.0001, 0.000001, 0.0000001]

for fold in folds_struct:
    print("fold index content", fold)
    folds_struct_filtered_ = [i for i in fold if mask[i]]
    print("fold filtered index content", folds_struct_filtered_)
    folds_struct_filtered.append(list(folds_struct_filtered_))

for file in files:
    # load MMD matrix
    X = np.loadtxt(os.path.join(directory, file))
    filename = os.path.splitext(file)[0]
    X_masked = X[mask]
    bootstrap_scores = []  # I will use only for uncertainty estimation in Accuracy

    # Applying bootstrapping for each file (crating list bootstrap iteration)
    # This is only for uncertainty estimation on the R^2 value from line 83 to 164
    for _ in range(n_bootstrap):
        models = []
        outputs = []
        scores = []
        y_total_test_masked = []

        # Applying bootstrapping for each list in folds_struct
        folds_struct_bootstrap = []
        best_alphas_per_fold_b = {}   # Dictionary to store the best alpha for each fold
        best_models_per_fold_b = {}   # Dictionary to store the best model for each fold
        best_outputs_per_fold_b = {}  # Dictionary to store the best outputs for each fold
        best_y_test_per_fold_b = {}   # Dictionary to store the corresponding true labels for each fold
        best_scores_per_fold_b = {}   # Dictionary to store the best score for each fold
        print("bootstrap iteration", _)
        # For uncertainty estimation in r^2 using bootstrapping
        for fold_list in folds_struct_filtered:
            indices = np.random.choice(fold_list, size=len(fold_list), replace=True)
            fold_bootstrap = indices
            folds_struct_bootstrap.append(list(fold_bootstrap))

        # Iterating over folds_struct
        for fold, fold_pre in zip(folds_struct_bootstrap, folds_struct_filtered):

            best_alpha_b = None
            best_score_alpha_b = float('-inf')  # Initialize with a very low score
            best_model_b = None
            best_outputs_b = None
            best_y_test_b = None

            fold_masked = fold    # from boostraped filtered fold
            fold_pre_ = fold_pre  # from fold filtered fold

            # index definition
            test_indices = fold_masked
            train_indices = [i for i in indices_mask if i not in fold_masked]

            # Training data:
            X_train_masked = X[train_indices][:, train_indices]
            data_train_masked = data[train_indices]

            # testing data:
            x_test_masked = X[test_indices][:, train_indices]
            y_test_masked = data[test_indices]
            
            for alpha in alpha_values:
                print(f"Training models for alpha = {alpha}")
                # Create and fit the model
                model2 = KernelRidge(alpha=alpha, kernel='rbf', gamma=0.1).fit(X_train_masked, data_train_masked)

                # Make predictions on the test set
                outputs_fold = model2.predict(x_test_masked)
        
                # Calculate the R2 score for the current alpha
                r2_alpha = r2_score(y_test_masked, outputs_fold)
        
                # Update best_alpha and best_score_alpha if necessary
                if r2_alpha > best_score_alpha_b:
                    best_alpha_b = alpha
                    best_score_alpha_b = r2_alpha
                    best_model_b = model2  # Save the model for the best alpha
                    best_outputs_b = outputs_fold
                    best_y_test_b = y_test_masked
                    best_scores_b = r2_score(best_outputs_b, best_y_test_b)
        
            # Store the best alpha, best model, and best outputs for this fold
            best_alphas_per_fold_b[tuple(fold)] = best_alpha_b  # Convert fold to a tuple
            best_models_per_fold_b[tuple(fold)] = best_model_b
            best_outputs_per_fold_b[tuple(fold)] = best_outputs_b
            best_y_test_per_fold_b[tuple(fold)] = best_y_test_b
            best_scores_per_fold_b[tuple(fold)] = best_scores_b
        
        # Now, use the best alphas and models to calculate the final R2 score and best model
        combined_outputs_b = np.concatenate(list(best_outputs_per_fold_b.values()))
        combined_y_test_b = np.concatenate(list(best_y_test_per_fold_b.values()))
        
        r2_final = r2_score(combined_y_test_b, combined_outputs_b)    
        bootstrap_scores.append(r2_final)

    # Model training
    print("Now Training on data:")
    i = 0
    models2 = []
    outputs2 = []
    scores2 = []
    y_total_test_masked2 = []

    best_alphas_per_fold = {}  # Dictionary to store the best alpha for each fold
    best_models_per_fold = {}  # Dictionary to store the best model for each fold
    best_outputs_per_fold = {}  # Dictionary to store the best outputs for each fold
    best_y_test_per_fold = {}  # Dictionary to store the corresponding true labels for each fold
    best_scores_per_fold = {}  # Dictionary to store the best score for each fold

    for fold in folds_struct_filtered:
        fold_masked = fold
        train_indices = [i for i in indices_mask if i not in fold_masked]
        test_indices = fold_masked

        # Training data:
        X_train_masked2 = X[train_indices][:, train_indices]
        data_train_masked2 = data[train_indices]

        # Testing data:
        x_test_masked2 = X[test_indices][:, train_indices]
        y_test_masked2 = data[test_indices]

        best_alpha = None
        best_score_alpha = float('-inf')  # Initialize with a very low score
        best_model = None
        best_outputs = None
        best_y_test = None

        for alpha in alpha_values:
            print(f"Training models for alpha = {alpha}")
            print("Actual folds")

            # Create and fit the model
            model2 = KernelRidge(alpha=alpha, kernel='rbf').fit(X_train_masked2, data_train_masked2)

            # Make predictions on the test set
            outputs_fold = model2.predict(x_test_masked2)

            # Calculate the R2 score for the current alpha
            r2_alpha = r2_score(y_test_masked2, outputs_fold)

            # Update best_alpha and best_score_alpha if necessary
            if r2_alpha > best_score_alpha:
                best_alpha = alpha
                best_score_alpha = r2_alpha
                best_model = model2  # Save the model for the best alpha
                best_outputs = outputs_fold
                best_y_test = y_test_masked2
                best_scores = r2_score(best_outputs, best_y_test)
                print("best", best_scores)

        # Store the best alpha, best model, and best outputs for this fold
        best_alphas_per_fold[tuple(fold)] = best_alpha  # Convert fold to a tuple
        best_models_per_fold[tuple(fold)] = best_model
        best_outputs_per_fold[tuple(fold)] = best_outputs
        best_y_test_per_fold[tuple(fold)] = best_y_test
        best_scores_per_fold[tuple(fold)] = best_scores

    # Now, use the best alphas and models to calculate the final R2 score and best model
    combined_outputs = np.concatenate(list(best_outputs_per_fold.values()))
    combined_y_test = np.concatenate(list(best_y_test_per_fold.values()))
    combined_fluct = list(best_scores_per_fold.values())
    r2_final2 = r2_score(combined_y_test, combined_outputs)

    # Calculate other metrics (uncertainty, Pearson correlation, Spearman correlation)
    # uncertainty_score = np.std(combined_outputs)
    pearson_corr, _ = pearsonr(combined_y_test, combined_outputs)
    spearman_corr, _ = spearmanr(combined_y_test, combined_outputs)

    print("Result of the model:", r2_final2)

    # Save results of predictions to a text file:
    with open("output_files/KMER_prediction_" + filename + ".txt", "w") as f:
        for j in range(len(combined_y_test)):
            f.write(str(combined_y_test[j]) + "\t" + str(combined_outputs[j]) + "\n")

    print("Done Kernel Ridge Regression Analysis")
    print("result", r2_final2)

    # Save the R2-score and correlation results for this file
    with open("output_files/KMER_r2_" + str(filename) + ".txt", "w") as f:
        f.write(f"r^2 {r2_final2}\n")
        f.write(f"r^2 error {np.std(bootstrap_scores)}\n")
        f.write(f"Pearson Corr: {pearson_corr}\n")
        f.write(f"Spearman Corr: {spearman_corr}\n")

    print("Next File if any")
print("Done!")
