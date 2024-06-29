import numpy as np
from sklearn import metrics
from joblib import Parallel, delayed, cpu_count

def k(x, y, gamma=1.0):
    #return metrics.pairwise.rbf_kernel(x.reshape(-1, 1), y.reshape(-1, 1), gamma=gamma)
    return metrics.pairwise.linear_kernel(x.reshape(-1, 1), y.reshape(-1, 1))
    #return metrics.pairwise.polynomial_kernel(x.reshape(-1, 1), y.reshape(-1, 1), degree=2, gamma=gamma, coef0=0)

# Distance MMD

def MMD_eeg_spectr_optimized(H_X, H_Z, nm):
    N_X = np.round(H_X[:, 1] * nm)
    N_Z = np.round(H_Z[:, 1] * nm)

    K_XX = k(H_X[:, 0], H_X[:, 0])
    K_ZZ = k(H_Z[:, 0], H_Z[:, 0])
    K_XZ = k(H_X[:, 0], H_Z[:, 0])

    T1 = np.sum(N_X[:, None] * N_X[None, :] * K_XX)
    T2 = np.sum(N_Z[:, None] * N_Z[None, :] * K_ZZ)
    T12 = np.sum(N_X[:, None] * N_Z[None, :] * K_XZ)

    T = np.sqrt((1 / nm**2) * T1 + (1 / nm**2) * T2 - (2 / nm**2) * T12)
    return T

# Kernel with MMD distance defined in  MMD_eeg_spectr_optimized parallel
def kernel_mmd_parallel(X, Y=None, gamma=1.0):
    n_samples = X.shape[0]
    if Y is None:
        Y = X
    n_samples_Y = Y.shape[0]

    def calculate_row(i):
        H_X = np.column_stack((np.arange(len(X[i])), X[i]))
        row = np.zeros(n_samples_Y)
        for j in range(n_samples_Y):
            H_Z = np.column_stack((np.arange(len(Y[j])), Y[j]))
            diff_s = MMD_eeg_spectr_optimized(H_X, H_Z, nm=200)
            row[j] = np.exp(-gamma * diff_s * diff_s)
        return row

    K = Parallel(n_jobs=cpu_count())(delayed(calculate_row)(i) for i in range(n_samples))
    return np.array(K)
