########################################################################
#      C.Jarne- 2023 Analysis Group D. Vidahurre  @cfin                #
#                                                                      #
# Estimation of the Maximum Mean Discrepancy (MMD) for                 #
# each individual with respect to a list. It is used to compute        #
# the distance matrix for each kernel in a set of individuals called   #
# from main_eeg_for_combined.py. Use the empirical MMD definitions for #
#  each kernel in the mmd_def.py library.                              #
########################################################################

import gc
from mmd_def import *

# MMD rbf distance with respect to the data_subjet_x
# MMD linear distance with respect to the data_subjet_x
# MMD polinomial distance with respect to the data_subjet_x
# MMD cos distance with respect to the data_subjet_x

def mmd_estimation(list_x, data_subjet_x):
    # parameters:
    # list made of distribution samples of each distribution of each sim individual
    # distribution samples for the reference indidìvidual (real data)
    
    drbf =[]
    dlinear = []
    dpoly = []
    dcosi = []

    for i in list_x:
       # data subject x is fix
       # i is each element in the list

       dpoly.append(mmd_poly(data_subjet_x,i)) 
       dcosi.append(mmd_cosi(data_subjet_x, i)) 
       drbf.append(mmd_rbf(data_subjet_x,i)) 
       dlinear.append(mmd_linear(np.array(data_subjet_x),np.array(i)))

    print("List with polinomial MMD with respect to target x:", dpoly)
    print("List with rbf MMD with respect to target x:", drbf)
    print("List with cosin MMD with respect to target x:", dcosi)
    print("List with linear MMD with respect to target x:", dlinear)
    return(dpoly,dcosi, drbf, dlinear)

gc.collect()
