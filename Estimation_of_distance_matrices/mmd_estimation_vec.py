# to save the distances
import gc

from mmd_def import *

# MMD rbf distance with respect to the data_subjet_x
# MMD linear distance with respect to the data_subjet_x
# MMD plinomial distance with respect to the data_subjet_x
# MMD cos distance with respect to the data_subjet_x


def mmd_estimation(list_x, data_subjet_x):
    #parameters:
    # list made of distribution samples of each distribution of each sim individual
    # distribution samples for the reference indididual (real data)
    
    drbf   =[]
    dlinear=[]
    dpoly  =[]
    dcosi  =[]

    for i in list_x:
  
       #if you want to use the samples of the empirical amplitude distribution
       
       #data subject x is fix
       #i is each element in the list
       """
       dpoly.append(mmd_poly(data_subjet_x,i)) 
       dcosi.append(mmd_cosi(data_subjet_x,i)) 
       drbf.append(mmd_rbf(data_subjet_x,i)) 
       dlinear.append(mmd_linear(data_subjet_x,i))
       
       """
       #if I want to compare time series directly vector vs vector of time series or any series such frequency
       # of each sim respect to time series of the reference real data
       
       #print("data_subjet_x",data_subjet_x)
       #print("i",i)
       
       """
       dpoly.append(mmd_poly([data_subjet_x],[i])) 
       dcosi.append(mmd_cosi([data_subjet_x], [i])) 
       drbf.append(mmd_rbf([data_subjet_x],[i])) 
       dlinear.append(mmd_linear(np.array([data_subjet_x]),np.array([i])))
       """
       dpoly.append(mmd_poly(data_subjet_x,i)) 
       dcosi.append(mmd_cosi(data_subjet_x, i)) 
       drbf.append(mmd_rbf(data_subjet_x,i)) 
       dlinear.append(mmd_linear(np.array(data_subjet_x),np.array(i)))
       
      
    print("List with polinomial MMD with respect to target x:",dpoly)
    print("List with rbf MMD with respect to target x:", drbf)
    print("List with cosin MMD with respect to target x:", dcosi)
    print("List with linear MMD with respect to target x:", dlinear)
    return(dpoly,dcosi, drbf, dlinear)
    
"""  
def mmd_error_estimation(list_x, data_subjet_x):

    
    drbf_e   =[]
    dlinear_e=[]
    dpoly_e  =[]
    dcosi_e  =[]
    
    for i in list_x:
        dpoly_e.append(mmd_poly_err([data_subjet_x],[i])) 
        dcosi_e.append(mmd_poly_err([data_subjet_x],[i])) 
        drbf_e.append(mmd_poly_err([data_subjet_x],[i])) 
        dlinear_e.append(mmd_poly_err([data_subjet_x],[i])) 
                
    print("List with polinomial MMD errors with respect to target x:",dpoly_e)   
    return(dpoly_e,dcosi_e, drbf_e, dlinear_e)
"""
gc.collect()

