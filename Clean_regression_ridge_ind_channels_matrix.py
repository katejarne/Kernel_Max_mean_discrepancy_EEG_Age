########################################################################################
#              C. Jarne 2023 Analysis Group of D. Vidahurre  @cfin                     #
#                                                                                      #
# Code performs a Kernel Ridge regression (called KRR in the paper) on a dataset which #
# is frequency histograms of EEG for each individual and channel. It kernelizes the    #
# channel content. It evaluates the performance of the models, generates plots to      #
# visualize the results, and saves the outcomes to txt files. Additionally, it employs #
# bootstrapping for estimating uncertainties in the accuracy metric.                   #                            #
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error,  mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.kernel_ridge import KernelRidge

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
n_bootstrap = 5
current_directory = os.path.dirname(__file__)

# Feature file
path = current_directory + "/input_files/Feature_file/"

# Spectrum directory
path2 = current_directory + "/input_files/individual_raw_spectrum/"
print(path)
print(path2)

def accuracy_estimation(k):
    ids = []
    ages = []
    x = []
    y = []

    # Read feature file ".txt"
    with open(path + "features_ok.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            if parts[3] != 'nan':
                ids.append(parts[0])
                ages.append(float(parts[3]))
                y.append(float(parts[3]))

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
                    xx.append(float(parts[k]))  # channel k

            xx_nor=xx/np.sum(xx)
            x.append(xx_nor)    # I use as a feature the normalized full histogram per channel

    y = np.array(y)
    x = np.array(x)
    x_d = x

    # Create a boolean mask to select only the values of "data" greater than or equal to 5
    mask = (y >= 5)

    folds_struct_filtered = []
    yy = y
    y = y[mask]
    x_d = x_d[mask]
    indices_mask = np.where(mask)[0]
    print("index len i. e. number of individuals that match the criteria: ", len(indices_mask))
    y = np.array(y)
    x_d = np.array(x_d).reshape(len(x_d), -1)

    alpha_values = [10, 100, 1, 0.3, 0.04, 2, 10, 0.5, 0.1, 0.001]  # RR

    best_alphas_per_fold = {}  # Dictionary to store the best alpha for each fold
    best_models_per_fold = {}  # Dictionary to store the best model for each fold
    best_outputs_per_fold = {}  # Dictionary to store the best outputs for each fold
    best_y_test_per_fold = {}  # Dictionary to store the corresponding true labels for each fold
    best_scores_per_fold = {}   # Dictionary to store the best score for each fold

    best_alphas_per_fold_b = {}  # Dictionary to store the best alpha for each fold
    best_models_per_fold_b = {}  # Dictionary to store the best model for each fold
    best_outputs_per_fold_b = {}  # Dictionary to store the best outputs for each fold
    best_y_test_per_fold_b = {}  # Dictionary to store the corresponding true labels for each fold
    best_scores_per_fold_b = {}   # Dictionary to store the best score for each fold

    for fold in folds_struct:
        print("fold", fold)
        folds_struct_filtered_ = [i for i in indices_mask if (i in fold)]  # [i for i in fold if mask[i] ]
        print("fold filtered", folds_struct_filtered_)
        folds_struct_filtered.append(list(folds_struct_filtered_))
        # Select only the values of "data" and "X" that meet the mask condition

    # I will use only for uncertainty estimation in Accuracy
    bootstrap_scores = []

    # Applying bootstrapping for each file (crating list bootstrap iteration)
    #########################################################################
    for _ in range(n_bootstrap):

        # Applying bootstrapping for each list in folds_struct
        folds_struct_bootstrap = []

        for fold_list in folds_struct_filtered:
            indices = np.random.choice(fold_list, size=len(fold_list), replace=True)
            fold_bootstrap = indices
            folds_struct_bootstrap.append(list(fold_bootstrap))

        # Iterating over folds_struct
        print("folds_struct_bootstrap",folds_struct_bootstrap)
        for fold, fold_pre in zip(folds_struct_bootstrap, folds_struct_filtered):

            fold_masked = fold   # from bootstrapped filtered fold
            fold_pre_ = fold_pre  # from fold filtered fold

            # index definition
            test_indices = fold_masked
            train_indices = [i for i in indices_mask if (i not in fold_masked or fold_pre_)]

            # Training data:
            X_train_masked = x[train_indices]
            data_train_masked = yy[train_indices]

            print("data_train_masked", len(data_train_masked))
            # testing data:
            x_test_masked = x[test_indices]
            y_test_masked = yy[test_indices]
            # print("y_test_masked",y_test_masked,len(y_test_masked))
            print("x_test_masked", len(x_test_masked))
            # Training the model on the boostraped data

            best_alpha_b = None
            best_score_alpha_b = float('-inf')  # Initialize with a very low score
            best_model_b = None
            best_outputs_b = None
            best_y_test_b = None

            for alpha in alpha_values:
                print(f"Training models for alpha = {alpha}")
                # Create and fit the model
                model_b = KernelRidge(alpha=alpha, kernel='rbf').fit(X_train_masked, data_train_masked)
                # Make predictions on the test set
                outputs_fold_b = model_b.predict(x_test_masked)

                # Calculate the R2 score for the current alpha
                r2_alpha_b = r2_score(y_test_masked, outputs_fold_b)

                # Update best_alpha and best_score_alpha if necessary
                if r2_alpha_b > best_score_alpha_b:
                    best_alpha_b = alpha
                    best_score_alpha_b = r2_alpha_b
                    best_model_b = model_b  # Save the model for the best alpha
                    best_outputs_b = outputs_fold_b
                    best_y_test_b = y_test_masked
                    best_scores_b = r2_score(best_outputs_b, best_y_test_b)
                    print("best", best_scores_b)

            # Store the best alpha, best model, and best outputs for this fold
            best_alphas_per_fold_b[tuple(fold)] = best_alpha_b  # Convert fold to a tuple
            best_models_per_fold_b[tuple(fold)] = best_model_b
            best_outputs_per_fold_b[tuple(fold)] = best_outputs_b
            best_y_test_per_fold_b[tuple(fold)] = best_y_test_b
            best_scores_per_fold_b[tuple(fold)] = best_scores_b
        # Now, use the best alphas and models to calculate the final R2 score and best model
        combined_outputs = np.concatenate(list(best_outputs_per_fold_b.values()))
        combined_y_test = np.concatenate(list(best_y_test_per_fold_b.values()))
        combined_fluct = list(best_scores_per_fold_b.values())
        r2_final2 = r2_score(combined_y_test, combined_outputs)
        print(combined_fluct)
        uncertainty_score = np.std(combined_fluct)
        print("Uncertainty", uncertainty_score)
        bootstrap_scores.append(r2_final2)

    #########################################################################
    print("Now Training on data:")

    for i, fold in enumerate(folds_struct):
        fold_masked = fold
        train_indices = [i for i in range(len(x_d)) if i not in fold_masked]
        test_indices = fold_masked
        print("fold index content", fold)
        print("fold filtered index content", fold_masked)

        # Seleccionar solo los valores que cumplen con la máscara y pertenecen al conjunto de entrenamiento
        x_train = x_d[train_indices]
        y_train = y[train_indices]
        test_indices = np.array(test_indices)
        test_indices = test_indices[test_indices < x_d.shape[0]]
        x_test = x_d[test_indices]
        y_test = y[test_indices]

        best_alpha = None
        best_score_alpha = float('-inf')  # Initialize with a very low score
        best_model = None
        best_outputs = None
        best_y_test = None

        for alpha in alpha_values:
            print(f"Training models for alpha = {alpha}")
            # Create and fit the model
            model2 = KernelRidge(alpha=alpha, kernel='rbf').fit(x_train, y_train)
            # Make predictions on the test set
            outputs_fold = model2.predict(x_test)

            # Calculate the R2 score for the current alpha
            r2_alpha = r2_score(y_test, outputs_fold)

            # Update best_alpha and best_score_alpha if necessary
            if r2_alpha > best_score_alpha:
                best_alpha = alpha
                best_score_alpha = r2_alpha
                best_model = model2  # Save the model for the best alpha
                best_outputs = outputs_fold
                best_y_test = y_test
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
    print(combined_fluct)
    uncertanty_score = np.std(combined_fluct)
    print("Uncertainty", uncertanty_score)
    r2_mean = r2_score(combined_y_test, combined_outputs)

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
    ax.set_title('Age prediction with RR Channel: '+str(k))
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1,label="\n RR RMSE: "+str(mse__)+" MEA: "+str(mea__)+"\n R spearman : "+str(corr_))
    plt.scatter(combined_y_test, combined_outputs, alpha=0.5, color="blue")
    plt.plot(combined_y_test,  slope*np.array(combined_y_test) + intercept, 'grey', alpha=0.75)
    plt.xlabel('Real age')
    plt.ylabel('Predicted age')
    plt.ylim([-10, 100])
    plt.xlim([0, 102])
    plt.legend(loc='upper left')
    figname = "plots/KRR_pred_vs_real_"+str(k)+"_eeg.png"
    fig.tight_layout()
    plt.savefig(figname, dpi=200)

    res2 = stats.spearmanr(combined_y_test, (combined_outputs-combined_y_test))
    corr2 = res2[0]
    corr2_ = "%.4f" % corr2

    # plot y_pred - y_test vs. y_test
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_title('Delta in age prediction with RR Channel: '+str(k))
    slope, intercept, r_value, p_value, std_err = stats.linregress(combined_y_test, combined_outputs - combined_y_test)
    plt.scatter(combined_y_test, np.array(combined_outputs) - combined_y_test, color="cyan", alpha=0.75)
    plt.plot(combined_y_test,  slope*np.array(combined_y_test) + intercept, 'grey',alpha=0.75, label="R spearman : "+str(corr2_))
    plt.ylim([-100, 100])
    plt.xlabel("Real age")
    plt.ylabel("(Predicted - Real) age")
    plt.axhline(0, color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    figname = "plots/KRR_error_"+str(k)+"r_eeg.png"
    fig.tight_layout()
    plt.savefig(figname,dpi=200)

    with open("output_files/KRR_mae_"+str(k)+".txt", "w") as file:
        file.write("MAE" + "\t" + str(mea_) + "\n")

    with open("output_files/KRR_r2_com_"+str(k)+".txt", "w") as file:
        file.write(f"r^2 {r2_mean}\n")
        file.write(f"r^2 error {np.std(bootstrap_scores)}\n")
        file.write(f"\Pearson Corr: {corr}\n")
        file.write(f"{res}\n")

    # Calculate the variance of the model
    accuracy = r2_mean
    return accuracy

# Calculate the average variance of the model from  0 to 18 channel
channels = list(np.arange(0,17,1))
channels.append(18)
resultados = []

for k in channels:
    accuracy = accuracy_estimation(k)
    resultados.append((k, accuracy))

# Generate output table
with open("output_files/KRR_r2_full.txt", "w") as f:
    f.write("k\tR2 per channel\n")
    for k, varianza_promedio in resultados:
        f.write(f"{k}\t{varianza_promedio}\n")

