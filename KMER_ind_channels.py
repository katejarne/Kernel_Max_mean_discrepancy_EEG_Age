####################################################################################################
#                   C. Jarne 2024 Analysis Group of D. Vidahurre  @cfin                            #
#                                                                                                  #
# This  Python code performs kernel ridge regression for EEG Distance matrices in directory (KMER) #
# The script starts by defining a list of folds index (which correspond to each site),             #
# then it reads the files from a directory, loads the target values and masks the data based       #
# on age or gender, then iterates over the files in the directory. For each file, it loads the     #
# MMD distance matrix and applies the mask, then iterates over the folds, splits the data          #
# into training and testing sets, trains a KRR model on the training data, predicts the target     #
# values for the testing data, and saves the model and the predicted values.                       #
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
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error,  mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold,LeaveOneGroupOut,StratifiedKFold
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from scipy import stats

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
# directory = current_directory + '/input_files/EEG_normalized_distance_matrices/'
# directory = current_directory + '/Estimation_of_distance_matrices/out/kernel_pol/'
directory = current_directory + '/Estimation_of_distance_matrices/out/kernel_gauss/'
# directory = current_directory + '/Estimation_of_distance_matrices/out/kernel_lineal/'
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Load target values (Feature file)

data_ = np.loadtxt(current_directory+'/input_files/Feature_file/features_ok.txt', delimiter='\t', usecols=3)
data = np.nan_to_num(data_)

gender = np.loadtxt(current_directory+'/input_files/Feature_file/features_ok.txt',
                    delimiter='\t', usecols=2, dtype=str)
print("gender", gender)

site = np.loadtxt(current_directory+'/input_files/Feature_file/features_ok.txt',
                  delimiter='\t', usecols=1, dtype=str)
print("site", site)

# Bolean mask to select age groups, gender or other features
mask = (data >= 5)  # &  (data <= 25)
# mask = (data >= 5)  & (gender == "['M']")# for gender sel (gender == "['F']")
# mask = (data >= 5) & (site == "NewYork")

# Select only the values of "data" and "X" that meet the mask condition
data_masked = data[mask]

# index that we used later for selection
indices_mask = np.where(mask)[0]
print("index len i. e. number of individuals that match the criteria: ", len(indices_mask))

folds_struct_filtered = []
results = []

# Hyperparameters to tune

param_grid = {'alpha': [100, 50, 30, 20, 5, 0.3, 0.04, 2, 10, 1, 0.5, 0.001, 0.03, 0.1, 0.05, 0.2,
                        0.7, 0.01, 0.0001, 0.000001, 0.0000001],
              'gamma': [0.0001, 0.001, 0.01, 0.05,0.5, 0.1, 1.0,
                        10.0, 20, 30, 50, 100, 150, 200, 500, 1000, 100000], 'kernel': ['rbf']}

Model = KernelRidge()

for fold in folds_struct:
    print("Fold index content", fold)
    folds_struct_filtered_ = [i for i in fold if mask[i]]
    print("Fold filtered index content", folds_struct_filtered_)
    folds_struct_filtered.append(list(folds_struct_filtered_))

groups = []

for i, fold in enumerate(folds_struct_filtered):
    fold_masked = [idx for idx in fold if idx in indices_mask]
    groups.extend([i] * len(fold_masked))

# Convert to  numpy array
groups = np.array(groups)
print("Groups", groups)

for file in files:
    # load MMD matrix
    X = np.loadtxt(os.path.join(directory, file))
    filename = os.path.splitext(file)[0]
    X_masked = X[mask]
    bootstrap_scores = []  # I will use only for uncertainty estimation in Accuracy
    print("Processing File: ", filename)
    combined_y_test = []
    combined_outputs = []

    r2_scores = []
    predictions_per_fold = []

    # For random folds use the following:
    group_kfold_r = KFold(n_splits=10, shuffle=True)

    # For fixed folds use the following:
    group_kfold = GroupKFold(n_splits=10)

    results_list = []

    pattern = re.compile(r'matrix(\d+)')
    match = pattern.search(filename)
    if match:
        k = int(match.group(1))
    else:
        k = None

    #  For fixed folds use the following:
    inner_cv = GroupKFold(n_splits=4)
    grid_search = GridSearchCV(Model, param_grid, scoring="r2", cv=inner_cv, verbose=3)

    for i, (train_index, test_index) in enumerate(group_kfold.split(X_masked, data_masked, groups)):

    # For random folds use the following (for example for one site):
    # for i, (train_index, test_index) in enumerate(group_kfold_r.split(X_masked, data_masked, groups)):

        # Training data:
        X_train_masked2 = X_masked[train_index][:, train_index]
        data_train_masked2 = data_masked[train_index]

        # Testing data:
        x_test_masked2 = X_masked[test_index][:, train_index]
        y_test_masked2 = data_masked[test_index]

        # you can change it for randomfolds
        # grid_search = GridSearchCV(Model, param_grid, scoring="r2", verbose=3)
        # if random folds
        # grid_search.fit(X_train_masked2, data_train_masked2)

        # Fit the model with GridSearchCV
        # if group folds
        grid_search.fit(X_train_masked2, data_train_masked2, groups=groups[train_index])

        # Get the best parameters
        best_params = grid_search.best_params_
        best_kernel = best_params['kernel']
        best_alpha = grid_search.best_params_['alpha']

        if 'gamma' in grid_search.best_params_:
            best_gamma = grid_search.best_params_['gamma']
        else:
            best_gamma = None

        # use the parameters to fit in this fold

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

    # To plot lineal fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(combined_y_test, combined_outputs)

    # To plot prediction
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_title('Age prediction with RR Channel: ' + str(k))
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1,
                label="\n RR RMSE: "+str(mse__)+" MEA: "+str(mea__)+"\n R spearman : "+str(corr_))
    plt.scatter(combined_y_test, combined_outputs, alpha=0.5, color="blue")
    plt.plot(combined_y_test,  slope*np.array(combined_y_test) + intercept, 'grey', alpha=0.75)
    plt.xlabel('Real age')
    plt.ylabel('Predicted age')
    plt.ylim([-10, 100])
    plt.xlim([0, 102])
    plt.legend(loc='upper left')
    figname = "plots/KMER_pred_vs_real_" + str(k) + "_eeg.png"
    fig.tight_layout()
    plt.savefig(figname, dpi=200)

    res2 = stats.spearmanr(combined_y_test, (combined_outputs-combined_y_test))
    corr2 = res2[0]
    corr2_ = "%.4f" % corr2

    # plot y_pred - y_test vs. y_test
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_title('Delta in age prediction with RR Channel: ' + str(k))
    slope, intercept, r_value, p_value, std_err = stats.linregress(combined_y_test, combined_outputs - combined_y_test)
    plt.scatter(combined_y_test, np.array(combined_outputs) - combined_y_test, color="cyan", alpha=0.75)
    plt.plot(combined_y_test,  slope*np.array(combined_y_test) + intercept, 'grey',
             alpha=0.75, label="R spearman : "+str(corr2_))
    plt.ylim([-100, 100])
    plt.xlabel("Real age")
    plt.ylabel("(Predicted - Real) age")
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    figname = "plots/KMER_error_" + str(k) + "r_eeg.png"
    fig.tight_layout()
    plt.savefig(figname, dpi=200)

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
        f.write(f"r^2 error {r2_cv_fluctuation}\n")
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
