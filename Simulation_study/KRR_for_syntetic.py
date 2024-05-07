########################################################################################
#              C. Jarne 2024 Analysis Group of D. Vidahurre  @cfin                     #
#                                                                                      #
# Code performs a Kernel Ridge regression (called KRR in the paper) on a dataset       #
# It saves the outcomes to txt files.                                                  #
# input: individual spectrum dir                                                       #
# consist of freq histograms of EEG for each individual channels at path files .txt    #
#                                                                                      #
# output:                                                                              #
# KRR_error_mae_+..+.txt Mean Absolute error for CH xx                                 #
# KRR_r2_+...+.txt R2 score for each with Pearson/Sperman correlation coefficient      #
# KRR_prediction_+..+.txt with real vs predicted age                                   #
# KRR_r2_full.txt all R2 score of each CH                                              #
########################################################################################

import os
import re
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error,  mean_absolute_error
from scipy import stats
from sklearn.kernel_ridge import KernelRidge

# Definir la ruta del directorio actual
current_directory = os.path.dirname(__file__)

# Ruta del directorio de características
feature_path = os.path.join(current_directory, "input_files", "Feature_file")

# Lista de todos los subdirectorios en "synthetic-data-grid"
synthetic_data_grid_dirs = [d for d in os.listdir(os.path.join(current_directory, "synthetic-data-grid"))
                            if os.path.isdir(os.path.join(current_directory, "synthetic-data-grid", d))]


Model = KernelRidge()

# Hyperparameters to tune

param_grid = {'alpha': [100, 50, 30, 20, 5, 0.3, 0.04, 2, 10, 1, 0.5, 0.001, 0.03, 0.1, 0.05, 0.2,
                        0.7, 0.01, 0.0001, 0.000001, 0.0000001],
              'gamma': [0.0001, 0.001, 0.01, 0.05, 0.5, 0.1, 1.0,
                        10.0, 20, 30, 50, 100, 150, 200, 500, 1000, 100000], 'kernel': ['rbf']}


def accuracy_estimation(path2, subdir):
    ids = []
    ages = []
    x = []
    y = []
    print(path2)
    print(subdir)

    with open(current_directory + '/synthetic-data-grid/features_syntetic_' + subdir + '.txt', "r") as file:
        print(file)
        next(file)
        for line in file:
            print(line)
            parts = line.strip().split()
            if parts[1] != 'nan':
                ids.append(parts[0])
                ages.append(float(parts[1]))
                y.append(float(parts[1]))

    # read the files in the directory
    nombres_check = []

    for nombre in ids:
        filename = f"{nombre}.txt"
        filepath = os.path.join(path2, filename)
        print(filepath)

        if os.path.isfile(filepath):
            # Read the x and y values from the file
            xx = []
            nombres_check.append(nombre)
            with open(filepath, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    xx.append(float(parts[0]))

            xx_nor = xx
            x.append(xx_nor)    # I use as a feature the histogram
    y = np.array(y)
    x = np.array(x)
    x_d = x

    # Create a boolean mask to select only the values of "data" greater than or equal to 5
    mask = (y >= -2)
    y = y[mask]
    x_d = x_d[mask]
    indices_mask = np.where(mask)[0]
    print("index len i. e. number of individuals that match the criteria: ", len(indices_mask))
    y = np.array(y)
    x_d = np.array(x_d).reshape(len(x_d), -1)
    print("Here", y)

    group_kfold_r = KFold(n_splits=5, shuffle=True)

    fold_scores = []
    predicciones_por_fold = []
    combined_y_test = []
    combined_outputs = []
    results_list = []

    grid_search = GridSearchCV(Model, param_grid, scoring='r2', cv=group_kfold_r, verbose=3)

    for i, (train_idx, test_idx) in enumerate(group_kfold_r.split(x_d, y)):

        x_train, x_test = x_d[train_idx], x_d[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        combined_y_test.extend(y[test_idx])
        grid_search.fit(x_train, y_train)

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
        with open("output_files/KRR_r2_folds" + str(subdir) + ".txt", "w") as file:
            for result in results_list:
                file.write(f"Fold {result['fold']}:\n")
                file.write(f"Best parameters: {result['best_params']}\n")
                file.write(f"Score: {result['test_score']}\n\n")

        with open("output_files/KRR_r2_folds_" + str(subdir) + ".txt", "w") as file:
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

    combined_y_test = np.array(combined_y_test)
    combined_outputs = np.array(combined_outputs)
    print(r2_cv_fluctuation)
    uncertainty_score = np.std(r2_cv_fluctuation)
    print("Uncertainty", uncertainty_score)
    r2_value = r2_score(combined_y_test, combined_outputs)
    r2_result = r2_score(combined_y_test, combined_outputs)

    res = stats.spearmanr(combined_y_test, combined_outputs)
    print("Result of model", res)
    corr = res[0]
    corr_ = "%.4f" % corr

    mse_ = mean_squared_error(combined_y_test, combined_outputs, squared=False)
    mea_ = mean_absolute_error(combined_y_test, combined_outputs)

    with open("output_files/KRR_mae_" + str(subdir) + ".txt", "w") as file:
        file.write("MAE" + "\t" + str(mea_) + "\n")

    with open("output_files/KRR_r2_com_" + str(subdir) + ".txt", "w") as file:
        file.write(f"r^2 {r2_value}\n")
        # file.write(f"r^2 error {np.std(bootstrap_scores)}\n")
        file.write(f"\Pearson Corr:{corr}\n")
        file.write(f"{res}\n")

    # Calculate the variance of the model

    print("Accuracy per channel", r2_result)
    r2_value
    return r2_result

results =[]

for sub_dir in synthetic_data_grid_dirs:
    print("Processing subdirectory: ", sub_dir)
    # Definir la ruta del directorio de datos sintéticos actual
    path2 = os.path.join(current_directory, "synthetic-data-grid", sub_dir)
    # Llamar a la función accuracy_estimation para realizar el análisis en el subdirectorio actual
    accuracy = accuracy_estimation(path2, sub_dir)  # Asegúrate de pasar el canal adecuado aquí
    results.append((sub_dir, accuracy))

# Generate output table
with open("output_files/full_KRR_r2.txt", "w") as f:
    f.write("k\tR2 per channel\n")
    for k, varianza_promedio in results:
        f.write(f"{k}\t{varianza_promedio}\n")
