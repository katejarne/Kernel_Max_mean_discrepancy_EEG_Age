########################################################################################
#              C. Jarne 2024 Analysis Group of D. Vidaurre  @cfin                      #
#                                                                                      #
# Code performs a Kernel Ridge regression (called KRR in the paper) or KMER on a       #
# dataset which consist of frequency histograms (EEG for each individual and channel). #
# It kernelises the channel content. It evaluates the performance of the models,       #
# generates plots to visualize the results, and saves the outcomes to txt files.       #
# input: individual spectrum dir                                                       #
# consist of freq histograms of EEG for each individual channels at path files .txt    #
#                                                                                      #
# output:                                                                              #
# KRR_error_mae_+..+.txt Mean Absolute error for CH xx                                 #
# KRR_r2_+...+.txt R2 score for each with Pearson/Sperman correlation coefficient      #
# KRR_prediction_+..+.txt with real vs predicted age                                   #
# KRR_r2_full.txt all R2 score of each CH                                              #
# Plot for predicted vs real age                                                       #
# Plot of delta (predicted -real) age vs real age                                      #
########################################################################################
import os
import numpy as np
from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn import metrics
import time
import matplotlib.pyplot as plt
from scipy import stats
from mmd_definition import *

# Chose between KMER or KRR
method = "KMER"
output_dir = "output_files"

folds_struct = [
    list(range(0, 40)), list(range(40, 86)), list(range(86, 359)),
    list(range(359, 492)), list(range(492, 791)), list(range(791, 989)),
    list(range(989, 1003)), list(range(1003, 1206)), list(range(1206, 1261)),
    list(range(1261, 1331)), list(range(1331, 1359)), list(range(1359, 1380)),
    list(range(1380, 1756)), list(range(1756, 1966))
]

current_directory = os.path.dirname(__file__)
path = current_directory + "/input_files/Feature_file/"
path2 = current_directory + "/input_files/individual_raw_spectrum/"

#  Kernel with Euclidan Distance

def rbf_kernel_euclidean(X, gamma):
    # Kernel using Euclidean distance (same as rbf_kernel_)
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            # Calculate the squared Euclidean distance between X[i] and X[j]
            diff = np.linalg.norm(X[i] - X[j]) ** 2
            # Assign the RBF kernel value to the K matrix
            K[i, j] = np.exp(-gamma * diff)

    return K

# Kernel with MMD distance defined in  MMD_eeg_spectr_optimized

def kernel_mmd(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):

            H_X = np.column_stack((np.arange(len(X[i])), X[i]))
            H_Z = np.column_stack((np.arange(len(X[j])), X[j]))
            # Calculates the distances
            diff_s = MMD_eeg_spectr_optimized(H_X, H_Z, nm=200)
            K[i, j] = np.exp(-gamma * diff_s*diff_s) # for kernel we used the squared distance

    return K

def custom_kernel_ridge_regression(K_train, y_train, alpha):
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(K_train, y_train)
    return ridge

def custom_predict(ridge_model, K_test):
    return ridge_model.predict(K_test)

def accuracy_estimation(channel):
    start_time = time.time()
    ids, x, y, site, gender = [], [], [], [], []
    # id, channel spectrum, age, site, gender
    with open(path + "features_ok.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            if parts[3] != 'nan':
                ids.append(parts[0])
                y.append(float(parts[3]))
                site.append(parts[1])
                gender.append(parts[2])

    nombres_check = []

    for nombre in ids:
        filename = f"{nombre}.txt"
        filepath = os.path.join(path2, filename)
        if os.path.isfile(filepath):
            xx = []
            nombres_check.append(nombre)
            with open(filepath, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    xx.append(float(parts[channel]))

            xx_nor = xx / np.sum(xx)  # np.log(xx)/np.sum(np.log(xx)) #Log of Spectrum
            x.append(xx_nor)

    y = np.array(y)
    x = np.array(x)
    site = np.array(site)
    gender = np.array(gender)

    # Filter individuals younger than ... or apply other filters
    # mask = (y >= 5)  & (gender == "['M']")# for gender sel (gender == "['F']")
    # mask = (y >= 5) & (site == "NewYork")
    mask = (y >= 5)
    y = y[mask]
    x = x[mask]
    site = site[mask]
    gender = gender[mask]

    # Filter indexes
    indices_mask = np.where(mask)[0]

    print("index len i. e. number of individuals that match the criteria: ", len(indices_mask))
    y = np.array(y)  # Log model np.log(np.array(y))
    x = np.array(x).reshape(len(x), -1)
    print(x.shape)

    groups = []
    for i, fold in enumerate(folds_struct):
        fold_masked = [idx for idx in fold if idx in indices_mask]
        groups.extend([i] * len(fold_masked))

    # Convert to numpy array

    groups = np.array(groups)
    print("Groups", groups)

    outer_cv = GroupKFold(n_splits=10)
    group_kfold_r = KFold(n_splits=10)
    errors, maes, r2s = [], [], []

    # Grid parameters
    param_grid_alpha = [10, 1, 0.5, 0.001, 0.03, 0.1, 0.05, 0.2, 0.7, 0.01]
    param_grid_gamma = [0.1, 1, 10]

    precomputed_kernels = {}

    # We estimate Kernel for the different gamma values
    for gamma in param_grid_gamma:

        print(f"Precomputing kernel matrix for gamma={gamma}...")
        if method == "KRR":
            precomputed_kernels[gamma] = rbf_kernel_euclidean(x, gamma=gamma)
        if method == "KMER":
            precomputed_kernels[gamma] = kernel_mmd(x, gamma=gamma)

        print(precomputed_kernels[gamma])
        print("Done kernel matrix")

    combined_outputs = []
    combined_y_test = []
    fold_scores = []
    results_list = []
    combined_genders = []
    combined_sites = []

    #  for random folds use:
    #for i, (train_idx, test_idx) in enumerate(group_kfold_r.split(x, y)):

    # Cross fold validation
    for train_idx, test_idx in outer_cv.split(x, y, groups):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        inner_cv = GroupKFold(n_splits=4)
        #inner_cv = KFold(n_splits=4)
        best_alpha, best_gamma, best_score = None, None, float('-inf')

        # Searching best alpha and gamma for this fold
        for gamma in param_grid_gamma:
            K_full = precomputed_kernels[gamma]
            K_train = K_full[np.ix_(train_idx, train_idx)]

            grid_search_alpha = RandomizedSearchCV(Ridge(fit_intercept=False),
                                                   param_distributions={'alpha': param_grid_alpha},
                                                   n_iter=10, scoring='r2', cv=inner_cv, verbose=3,
                                                   n_jobs=-1)
            grid_search_alpha.fit(K_train, y_train, groups=groups[train_idx])

            if grid_search_alpha.best_score_ > best_score:
                best_alpha = grid_search_alpha.best_params_['alpha']
                best_gamma = gamma
                best_score = grid_search_alpha.best_score_

        print(f"Best alpha for this fold: {best_alpha}")
        print(f"Best gamma for this fold: {best_gamma}")
        K_train = precomputed_kernels[best_gamma][np.ix_(train_idx, train_idx)]
        ridge_model = custom_kernel_ridge_regression(K_train, y_train, alpha=best_alpha)

        # Fitting best model in this fold
        ridge_model.fit(K_train, y_train)

        # Testing model and saving predictions of current fold
        K_test = precomputed_kernels[best_gamma][np.ix_(test_idx, train_idx)]
        y_pred = custom_predict(ridge_model, K_test)

        mae_fold = mean_absolute_error(y_test, y_pred)
        maes.append(mae_fold)

        # To fill output file
        combined_outputs.extend(y_pred)
        combined_y_test.extend(y_test)
        combined_genders.extend(gender[test_idx])
        combined_sites.extend(site[test_idx])
        fold_score = r2_score(y_test, y_pred)
        fold_scores.append(fold_score)

        print(f"Inner R2 score for fold {i}: {fold_score}")
        fold_results = {'fold': i, 'best_params': {'alpha': best_alpha, 'gamma': best_gamma}, 'test_score': fold_score}
        results_list.append(fold_results)

    # Model with all folds
    r2_cv_fluctuation = np.std(fold_scores)
    mae = mean_absolute_error(combined_y_test, combined_outputs)
    r2 = r2_score(combined_y_test, combined_outputs)

    # Profiling
    elapsed_time = time.time() - start_time

    # Saving results:
    with open(f"{output_dir}/{method}_r2_folds_{channel}.txt", "w") as file:
        for result in results_list:
            file.write(f"Fold {result['fold']}:\n")
            file.write(f"Best parameters: {result['best_params']}\n")
            file.write(f"Score: {result['test_score']}\n\n")

    with open(f"{output_dir}/{method}_prediction_{channel}.txt", "w") as file:
        for actual, predicted, gend, sites in zip(combined_y_test, combined_outputs, combined_genders, combined_sites):
            file.write(f"{actual}\t{predicted}\t{gend}\t{sites}\n")

    combined_y_test = np.array(combined_y_test)
    combined_outputs = np.array(combined_outputs)
    res = stats.spearmanr(combined_y_test, combined_outputs)
    print("Result of model", res)
    corr = res[0]
    corr_ = "%.4f" % corr

    mse_ = mean_squared_error(combined_y_test, combined_outputs, squared=False)
    mea_ = mean_absolute_error(combined_y_test, combined_outputs)
    mse__ = "%.2f" % mse_
    mea__ = "%.2f" % mea_

    # To plot lineal fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(combined_y_test, combined_outputs)

    # To plot prediction
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_title(f'Age prediction with {method} Channel: ' + str(channel))
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1,
                label=f"{method} MEA: "+str(mea__)+"\n R: "+str(corr_))
    if method == "KMER":
        plt.scatter(combined_y_test, combined_outputs, alpha=0.5, color="#5ec962")
    if method == "KRR":
        plt.scatter(combined_y_test, combined_outputs, alpha=0.5, color="pink")
    plt.plot(combined_y_test,  slope*np.array(combined_y_test) + intercept, 'grey', alpha=0.75)
    plt.ylim([-10, 100])
    plt.xlim([0, 102])
    plt.xticks(np.arange(0, 101, 10), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Age", fontsize=18)
    plt.ylabel("Predicted age", fontsize=18)
    plt.legend(loc='upper left')
    figname = f"plots/{method}_pred_vs_real_" + str(channel) + "_eeg.png"
    fig.tight_layout()
    plt.savefig(figname, dpi=200)
    plt.close(fig)

    res2 = stats.spearmanr(combined_y_test, (combined_outputs-combined_y_test))
    corr2 = res2[0]
    corr2_ = "%.4f" % corr2

    # plot y_pred - y_test vs. y_test
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_title(f'Delta in age prediction with {method} Channel: ' + str(channel))
    slope, intercept, r_value, p_value, std_err = stats.linregress(combined_y_test, combined_outputs - combined_y_test)
    if method == "KMER":
        plt.scatter(combined_y_test, np.array(combined_outputs) - combined_y_test, color="#5ec962", alpha=0.75)
    if method == "KRR":
         plt.scatter(combined_y_test, np.array(combined_outputs) - combined_y_test, color="pink", alpha=0.75)
    plt.plot(combined_y_test,  slope*np.array(combined_y_test) + intercept, 'grey',
             alpha=0.75, label="R: "+str(corr2_))
    plt.ylim([-100, 100])
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel("Real age",fontsize=18)
    plt.ylabel("(Predicted - Real) age",fontsize=18)
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    figname = f"plots/{method}_error_" + str(channel) + "r_eeg.png"
    fig.tight_layout()
    plt.savefig(figname, dpi=200)
    plt.close(fig)

    with open(f"{output_dir}/{method}_mae_" + str(channel) + ".txt", "w") as file:
        file.write("MAE" + "\t" + str(mea_) + "\n")

    with open(f"{output_dir}/{method}_r2_com_" + str(channel) + ".txt", "w") as file:
        file.write(f"r^2 {r2}\n")
        file.write(f"r^2 error {r2_cv_fluctuation}\n")
        file.write(f"Pearson Corr:{corr}\n")
        file.write(f"{res}\n")

    return np.mean(maes), mae, r2, elapsed_time

# Calculate the average variance of the model from  0 to 18 channel
# channels = [0, 5, 12]
channels = list(np.arange(0, 17, 1))
channels.append(18)
results = []
channels = np.array(channels)
for ch in channels:
    print("Processing channel: ", ch)

    maez, mae, r2, elapsed_time = accuracy_estimation(ch)
    print(f"Mean Absolute Error average: {maez}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")
    print(f"Elapsed Time: {elapsed_time} seconds")
    results.append((ch, r2))

# Generate output table
with open(f"{output_dir}/full_{method}_r2.txt", "w") as f:
    f.write("k\tR2 per channel\n")
    for k, res in results:
        f.write(f"{k}\t{res}\n")
