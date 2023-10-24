######################################################################################
#               C.Jarne- 2023 Analysis Group D. Vidahurre  @cfin                     #
#                                                                                    #
# This code defines four functions for computing Maximum Mean Discrepancy (MMD)      #
# using different kernel functions such as linear, RBF, polynomial, and cosine. MMD  #
# is a distance metric used to measure the distance between two probability          #
# distributions. The function definitions have detailed documentation of their input #
# and output parameters. The main function tests the MMD functions with some sample  #
# input matrices and vectors and prints the results. The last line of the            #
# code prints the length of a vector.                                                #
######################################################################################

import numpy as np
from sklearn import metrics

# Function definition for Maximum Mean Discrepancy for the 4 different Kernel

# https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py

def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)

def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def mmd_cosi(X,Y):
    """MMD using cos kernel"""
    XX = metrics.pairwise.pairwise_kernels(X,X,metric="cosine")
    YY = metrics.pairwise.pairwise_kernels(Y,Y,metric="cosine")
    XY = metrics.pairwise.pairwise_kernels(X,Y,metric="cosine")
    return XX.mean() + YY.mean() - 2 * XY.mean()

if __name__ == '__main__':
    # Some tests
    a = np.arange(1, 10).reshape(3, 3)
    b = [ [4, 3, 2], [0, 2, 5],[1, 1, 8],[7, 6, 5]]
    b = np.array(b)
    print("a: ",a)
    print("b:", b)
    print(mmd_linear(a, b))  # 6.0
    print(mmd_rbf(a, b))     # 0.5822
    print(mmd_cosi(a,b))     # 0.024
    vector1 = np.array([[4, 3, 2]])
    vector2 = np.array([[2, 4, 1]])
    print(mmd_poly(a,b))
    print("len(vector1[0]", len(vector1[0]))
    print("d", mmd_poly(vector1,vector2))
