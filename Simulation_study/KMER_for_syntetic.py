####################################################################################################
#                   C. Jarne 2024 Analysis Group of D. Vidahurre  @cfin                            #
#                                                                                                  #
# This  Python code performs kernel ridge regression  Distance matrices in directory (KMER)        #
# The script reads the files from a directory, loads the target values and masks the data, then    #
# iterates over the files in the directory. For each file, it loads the                            #
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
import re
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error,  mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold
from scipy.stats import pearsonr, spearmanr
from scipy import stats

# Get the list of files in the directory
current_directory = os.path.dirname(__file__)
directory = current_directory + '/synthetic-data-grid/'
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

folds_struct_filtered = []
results = []

# Hyperparameters to tune

param_grid = {'alpha': [100, 50, 30, 20, 5, 0.3, 0.04, 2, 10, 1, 0.5, 0.001, 0.03, 0.1, 0.05, 0.2,
                        0.7, 0.01, 0.0001, 0.000001, 0.0000001],
              'gamma': [0.0001, 0.001, 0.01, 0.05, 0.5, 0.1, 1.0,
                        10.0, 20, 30, 50, 100, 150, 200, 500, 1000, 100000], 'kernel': ['rbf']}

Model = KernelRidge()

for file in files:
    print(file)
    # load MMD matrix
    if file.endswith('_distances.txt'):
        X = np.loadtxt(os.path.join(directory, file))
    else:
        continue
    filename = os.path.splitext(file)[0]
    print(filename)
    pattern = re.compile(r'(.+)_distances')
    match = pattern.match(filename)
    if match:
        part_before_distances = match.group(1)
    else:
        print("No match found")
    # Feature file corresponding to the distance matrix
    features_file = current_directory + '/synthetic-data-grid/features_syntetic_' +part_before_distances+ '.txt'
    # Load target values (feature file)
    print(features_file)
    data_ = np.loadtxt(features_file, delimiter='\t', usecols=1, skiprows=1)
    print(data_)
    data = np.nan_to_num(data_)
    # Bolean mask to select age groups, gender or other features
    mask = (data >= -2)

    # Select only the values of "data" and "X" that meet the mask condition
    data_masked = data[mask]

    # index that we used later for selection
    indices_mask = np.where(mask)[0]
    print("index len i. e. number of individuals that match the criteria: ", len(indices_mask))

    X_masked = X[mask]
    print("Processing File: ", filename)
    combined_y_test = []
    combined_outputs = []

    r2_scores = []
    predictions_per_fold = []

    # For random folds use the following:
    group_kfold_r = KFold(n_splits=5, shuffle=True)

    results_list = []

    pattern = re.compile(r'matrix(\d+)')
    match = pattern.search(filename)
    if match:
        k = part_before_distances
    else:
        k = None

    # grid_search = GridSearchCV(Model, param_grid, scoring="r2", cv=group_kfold_r, verbose=3)

    # For random folds use the following (for example for one site):
    for i, (train_index, test_index) in enumerate(group_kfold_r.split(X_masked, data_masked)):

        # Training data:
        X_train_masked2 = X_masked[train_index][:, train_index]
        data_train_masked2 = data_masked[train_index]

        # Testing data:
        x_test_masked2 = X_masked[test_index][:, train_index]
        y_test_masked2 = data_masked[test_index]

        grid_search = GridSearchCV(Model, param_grid, scoring="r2", verbose=3)
        grid_search.fit(X_train_masked2, data_train_masked2)

        # Get the best parameters
        best_params = grid_search.best_params_
        best_kernel = best_params['kernel']
        best_alpha = grid_search.best_params_['alpha']

        if 'gamma' in grid_search.best_params_:
            best_gamma = grid_search.best_params_['gamma']
        else:
            best_gamma = None

        # use this parameters to fit in this fold

        model = KernelRidge(alpha=best_alpha, gamma=best_gamma, kernel=best_kernel)
        model.fit(X_train_masked2, data_train_masked2)

        combined_y_test.extend(y_test_masked2)
        predictions_test = model.predict(x_test_masked2)
        combined_outputs.extend(predictions_test)
        r2_score_fold = r2_score(y_test_masked2, predictions_test)

        # Store the result for the fold
        r2_scores.append(r2_score_fold)

        print(f"Best alpha for this fold: {best_alpha}")
        print(f"Best gamma for this fold: {best_gamma}")
        print(f"R2 for this fold: {r2_score_fold}")

        # Save results in list
        fold_results = {'fold': i, 'best_params': grid_search.best_params_,
                        'test_score': r2_score_fold}
        results_list.append(fold_results)

        # Saving results of each fold
        with open("output_files/KMER_r2_folds" + str(k) + ".txt", "w") as file:
            for result in results_list:
                file.write(f"Fold {result['fold']}:\n")
                file.write(f"Best parameters: {result['best_params']}\n")

                file.write(f"Score: {result['test_score']}\n\n")

        with open("output_files/KMER_r2_folds_" + str(k) + ".txt", "w") as file:
            for r2 in r2_scores:
                file.write(f"r^2 {r2}\n")

    # Calculate the average R2 and standard deviation
    average_r2 = np.mean(r2_scores)
    r2_cv_fluctuation = np.std(r2_scores)

    print(f"Average R2: {average_r2}")
    print(f"Standard deviation of R2: {r2_cv_fluctuation}")
    uncertainty_score = r2_cv_fluctuation
    print("Uncertainty", uncertainty_score)
    print("Real age", combined_y_test)
    print("Predicted age", combined_outputs)
    combined_y_test = np.array(combined_y_test)
    combined_outputs = np.array(combined_outputs)
    uncertainty_score = np.std(r2_cv_fluctuation)

    r2_result = r2_score(combined_y_test, combined_outputs)

    # Calculate other metrics (Pearson correlation, Spearman correlation)

    pearson_corr, _ = pearsonr(combined_y_test, combined_outputs)
    spearman_corr, _ = spearmanr(combined_y_test, combined_outputs)

    res = stats.spearmanr(combined_y_test, combined_outputs)
    mse_ = mean_squared_error(combined_y_test, combined_outputs, squared=False)
    mea_ = mean_absolute_error(combined_y_test, combined_outputs)
    mse__ = "%.2f" % mse_
    mea__ = "%.2f" % mea_

    print("Result of model", res)
    corr = res[0]
    corr_ = "%.4f" % corr
    print("Result of the model:", r2_result)

    # Save results of predictions to a text file:
    with open("output_files/KMER_prediction_" + filename + ".txt", "w") as f:
        for j in range(len(combined_y_test)):
            f.write(str(combined_y_test[j]) + "\t" + str(combined_outputs[j]) + "\n")

    with open("output_files/KMER_mae_" + str(filename) + ".txt", "w") as file:
        file.write("MAE" + "\t" + str(mea_) + "\n")

    print("Done Kernel Ridge Regression Analysis")

    # Save the R2-score and correlation results for this file
    with open("output_files/KMER_r2_" + str(filename) + ".txt", "w") as f:
        f.write(f"r^2 {r2_result}\n")
        f.write(f"Pearson Corr: {pearson_corr}\n")
        f.write(f"Spearman Corr: {spearman_corr}\n")

    print("Next File if any")
    results.append((k, r2_result))

# Generate output table
with open("output_files/full_KMER_r2.txt", "w") as f:
    f.write("k\tR2 per channel\n")
    for k, result in results:
        f.write(f"{k}\t{result}\n")

print("Done!")
