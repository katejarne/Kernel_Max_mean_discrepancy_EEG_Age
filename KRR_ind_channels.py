########################################################################################
#              C. Jarne 2023 Analysis Group of D. Vidahurre  @cfin                     #
#                                                                                      #
# Code performs a Kernel Ridge regression (called KRR in the paper) on a dataset which #
# is frequency histograms of EEG for each individual and channel. It kernelizes the    #
# channel content. It evaluates the performance of the models, generates plots to      #
# visualize the results, and saves the outcomes to txt files. Additionally, it employs #
# bootstrapping for estimating uncertainties in the accuracy metric.                   #
#                                                                                      #
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
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error,  mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.kernel_ridge import KernelRidge

folds_struct = [
    list(range(0, 40)),       # Chengdu
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
n_bootstrap = 5
current_directory = os.path.dirname(__file__)

# Feature file
path = current_directory + "/input_files/Feature_file/"

# Spectrum directory

path2 = current_directory + "/input_files/individual_raw_spectrum/"
print(path)
print(path2)

# Model = KernelRidge(kernel="linear")
Model = KernelRidge()

# Hyperparameters to tune

param_grid = {'alpha': [100, 50, 30, 20, 5, 0.3, 0.04, 2, 10, 1, 0.5, 0.001, 0.03, 0.1, 0.05, 0.2,
                        0.7, 0.01, 0.0001, 0.000001, 0.0000001],
              'gamma': [0.0001, 0.001, 0.01, 0.05,0.5, 0.1, 1.0,
                        10.0, 20, 30, 50, 100, 150, 200, 500, 1000, 100000], 'kernel': ['rbf']}


def accuracy_estimation(channel):
    ids = []
    ages = []
    x = []
    y = []
    site = []
    gender = []

    # Read feature file ".txt"
    with open(path + "features_ok.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            if parts[3] != 'nan':
                ids.append(parts[0])
                ages.append(float(parts[3]))
                y.append(float(parts[3]))
                site.append(parts[1])
                gender.append((parts[2]))

    # read the files in the directory
    nombres_check = []

    for nombre in ids:
        filename = f"{nombre}.txt"
        filepath = os.path.join(path2, filename)
        if os.path.isfile(filepath):
            # Read the x and y values from the file
            xx = []
            nombres_check.append(nombre)
            with open(filepath, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    xx.append(float(parts[channel]))  # channel k

            # xx_nor = np.log(xx)/np.sum(np.log(xx))
            xx_nor = xx/np.sum(xx)
            x.append(xx_nor)    # I use as a feature the normalized full histogram per channel

    y = np.array(y)
    x = np.array(x)
    x_d = x
    ages = np.array(ages)
    site = np.array(site)
    gender = np.array(gender)

    # Create a boolean mask to select only the values of "data" greater than or equal to 5
    mask = (y >= 5)   # & (y <= 25)
    # mask = (y >= 5) & (site == 'NewYork')
    # mask = (y >= 5) & (site == "CHBMP")
    # mask = (y >= 5) & (site == "Russia")
    # folds_struct_filtered = []
    # yy = y
    y = y[mask]
    x_d = x_d[mask]
    indices_mask = np.where(mask)[0]
    print("index len i. e. number of individuals that match the criteria: ", len(indices_mask))
    y = np.array(y)
    x_d = np.array(x_d).reshape(len(x_d), -1)

    groups = []
    # To follow the fold structure and obtain the 14 cv folds results
    for i, fold in enumerate(folds_struct):
        fold_masked = [idx for idx in fold if idx in indices_mask]
        groups.extend([i] * len(fold_masked))

    # Convert to  numpy array
    groups = np.array(groups)
    print("Groups", groups)

    # For random folds use the following:

    n_splits = 4
    # group_kfold_r = KFold(n_splits=n_splits, shuffle=True)

    # To respect the fold structure on the grid search

    group_kfold = GroupKFold(n_splits=10)
    # group_kfold_p= GroupKFold(n_splits=4)

    fold_scores = []
    combined_y_test = []
    combined_outputs = []
    results_list = []

    inner_cv = GroupKFold(n_splits=4)
    grid_search = GridSearchCV(Model, param_grid, scoring='r2', cv=inner_cv, verbose=3)

    for i, (train_idx, test_idx) in enumerate(group_kfold.split(x_d, y, groups)):
    # for random folds use:
    # for i, (train_idx, test_idx) in enumerate(group_kfold_r.split(x_d, y, groups)):

        x_train, x_test = x_d[train_idx], x_d[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        combined_y_test.extend(y[test_idx])

        grid_search.fit(x_train, y_train, groups=groups[train_idx])
        # grid_search.fit(x_train, y_train)

        best_params = grid_search.best_params_
        best_kernel = best_params['kernel']
        if 'gamma' in grid_search.best_params_:
            best_gamma = grid_search.best_params_['gamma']
        else:
            best_gamma = None

        # Get the best alpha value and calculate R2 on the test set
        best_alpha = grid_search.best_params_['alpha']
        best_gamma = grid_search.best_params_['gamma']

        model = KernelRidge(alpha=best_alpha, gamma=best_gamma, kernel=best_kernel)
        model.fit(x_train, y_train)
        predicciones_por_fold = model.predict(x_test)
        fold_score = r2_score(y_test, predicciones_por_fold)
        fold_scores.append(fold_score)
        combined_outputs.extend(predicciones_por_fold)

        print(f"Best alpha for this fold: {best_alpha}")
        print(f"Best gamma for this fold: {best_gamma}")
        print("Inner", fold_score)

        # Save results in list
        fold_results = {'fold': i, 'best_params': grid_search.best_params_,
                        'test_score': fold_score}

        results_list.append(fold_results)

        # Saving results of each fold
        with open("output_files/KRR_r2_folds" + str(channel) + ".txt", "w") as file:
            for result in results_list:
                file.write(f"Fold {result['fold']}:\n")
                file.write(f"Best parameters: {result['best_params']}\n")
                file.write(f"Score: {result['test_score']}\n\n")

        with open("output_files/KRR_r2_folds_" + str(channel) + ".txt", "w") as file:
            for r2 in fold_scores:
                file.write(f"r^2 {r2}\n")

    # Calculate the average R2 and standard deviation
    average_r2 = np.mean(fold_scores)
    r2_cv_fluctuation = np.std(fold_scores)

    print(f"Average R2: {average_r2}")
    print(f"Standard deviation of R2: {r2_cv_fluctuation}")
    print(r2_cv_fluctuation)
    uncertainty_score = r2_cv_fluctuation
    print("Uncertainty", uncertainty_score)
    print("Real age", combined_y_test)
    print("Predicted age", combined_outputs)
    combined_y_test = np.array(combined_y_test)
    combined_outputs = np.array(combined_outputs)
    print(r2_cv_fluctuation)
    uncertainty_score = r2_cv_fluctuation
    print("Uncertainty", uncertainty_score)
    r2_value = r2_score(combined_y_test, combined_outputs)
    r2_result = r2_score(combined_y_test, combined_outputs)

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
    ax.set_title('Age prediction with RR Channel: ' + str(channel))
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1,label="\n RR RMSE: "+str(mse__)+" MEA: "+str(mea__)+"\n R spearman : "+str(corr_))
    plt.scatter(combined_y_test, combined_outputs, alpha=0.5, color="blue")
    plt.plot(combined_y_test,  slope*np.array(combined_y_test) + intercept, 'grey', alpha=0.75)
    plt.xlabel('Real age')
    plt.ylabel('Predicted age')
    plt.ylim([-10, 100])
    plt.xlim([0, 102])
    plt.legend(loc='upper left')
    figname = "plots/KRR_pred_vs_real_" + str(channel) + "_eeg.png"
    fig.tight_layout()
    plt.savefig(figname, dpi=200)

    res2 = stats.spearmanr(combined_y_test, (combined_outputs-combined_y_test))
    corr2 = res2[0]
    corr2_ = "%.4f" % corr2

    # plot y_pred - y_test vs. y_test
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_title('Delta in age prediction with RR Channel: ' + str(channel))
    slope, intercept, r_value, p_value, std_err = stats.linregress(combined_y_test, combined_outputs - combined_y_test)
    plt.scatter(combined_y_test, np.array(combined_outputs) - combined_y_test, color="cyan", alpha=0.75)
    plt.plot(combined_y_test,  slope*np.array(combined_y_test) + intercept, 'grey',alpha=0.75, label="R spearman : "+str(corr2_))
    plt.ylim([-100, 100])
    plt.xlabel("Real age")
    plt.ylabel("(Predicted - Real) age")
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    figname = "plots/KRR_error_" + str(channel) + "r_eeg.png"
    fig.tight_layout()
    plt.savefig(figname, dpi=200)

    with open("output_files/KRR_mae_" + str(channel) + ".txt", "w") as file:
        file.write("MAE" + "\t" + str(mea_) + "\n")

    with open("output_files/KRR_r2_com_" + str(channel) + ".txt", "w") as file:
        file.write(f"r^2 {r2_value}\n")
        file.write(f"r^2 error {uncertainty_score}\n")
        file.write(f"Pearson Corr:{corr}\n")
        file.write(f"{res}\n")

    # Calculate the variance of the model
    print("Accuracy per channel", r2_result)
    r2_value
    return r2_result

# Calculate the average variance of the model from  0 to 18 channel
channels = list(np.arange(0, 17, 1))
channels.append(18)

#channels = [0]# [0, 5, 12]
results = []

for k in channels:
    print("Processing channel: ", k)
    # accuracy = accuracy_estimation(k)
    accuracy = accuracy_estimation(k)
    results.append((k, accuracy))

# Generate output table
with open("output_files/full_KRR_r2.txt", "w") as f:
    f.write("k\tR2 per channel\n")
    for k, varianza_promedio in results:
        f.write(f"{k}\t{varianza_promedio}\n")
