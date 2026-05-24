# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
     
     Prof. Francisco Assis de Oliveira Nascimento
     Digital Signal Processing Group - DSPG/ENE/FT/UnB
     Electrical Engineering Department
     Faculty of Technology
     University of Brasilia, Brazil
     
-------------------------------------------------------------------------------
"""
import numpy as np
import scipy
#from scipy.stats import skew
#from scipy.stats import kurtosis
import skimage
from skimage import feature, measure, data, color, exposure, io
import cv2
#------------------------------------------------------------------------------
# Pause program
#------------------------------------------------------------------------------
def pause():
    programPause = input("Press the <ENTER> key to continue...") 
#------------------------------------------------------------------------------   
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#--------------------------------------------------------------------------
#
#   Compute First Order Statistics.
#   FAON May 15, 2022.
#--------------------------------------------------------------------------
def Compute_First_Order_Statistics_Features(image, Bins):        
    features = []     
    N, M = image.shape                          
    histRange = (0, 256) # the upper boundary is exclusive.
    hist = cv2.calcHist(image, [0], None, [Bins], histRange) 
    histogram = hist.flatten()
    hist_sum = sum(histogram)
    #----------------------------------------------------------------------
    # Power Mass Function.
    #----------------------------------------------------------------------
    pmf = histogram / hist_sum                               
    #pmf = np.round(pmf, 3)
    #print('\n', pmf)        
    #----------------------------------------------------------------------
    image_min = np.min(image) 
    image_max = np.max(image) 
    image_mean = np.mean(image) 
    image_std = np.std(image) 
    image_var = np.var(image)
    #----------------------------------------------------------------------
    image_median = np.median(image)
    image_skew = scipy.stats.skew(image, axis = None)
    image_kurtpsis = scipy.stats.kurtosis(image, axis = None)
    square_range = (image_max - image_min)**2
    image_entropy = skimage.measure.shannon_entropy(image) 
    #print(image_entropy)
    #----------------------------------------------------------------------
    # Entropy 2 - it means the "bins entropy", as I called it.
    #----------------------------------------------------------------------
    epsilon = 2.2e-16
    bins_entropy = 0
    for i in range(Bins):
        bins_entropy -= pmf[i] * np.log2(pmf[i] + epsilon)
    #print (bins_entropy)
    #----------------------------------------------------------------------        
    # Normalized energy.
    #----------------------------------------------------------------------
    image_norm_energy = np.sum( image**2, axis = None ) / float(N * M)
    #----------------------------------------------------------------------
    # Root-Mean-Square (RMS) value.
    #----------------------------------------------------------------------
    image_rms = np.sqrt(image_norm_energy) 
    #----------------------------------------------------------------------
    # Mean Absolute Deviation.
    #----------------------------------------------------------------------
    image_abs_deviation = np.sum( np.abs(image - image_mean), axis = None ) / float(N * M)
    #----------------------------------------------------------------------
    # Percentiles.
    #----------------------------------------------------------------------
    image_p925 = np.percentile(image, 92.5)
    image_p850 = np.percentile(image, 85) 
    image_p75 = np.percentile(image, 7.5) 
    image_p150 = np.percentile(image, 15)
    interquartile_range = np.percentile(image, 75) - np.percentile(image, 25) 
    image_uniformity = np.sum( pmf**2, axis = None )         
    #----------------------------------------------------------------------
    # Compute the Robust Mean Absolute Deviation (rMAD).
    #----------------------------------------------------------------------
    image_p900 = np.percentile(image, 90.0)        
    image_p100 = np.percentile(image, 10.0)        
    Np_10_90 = 0 
    rMAD = 0       
    for n in range(N):
        for m in range(M):
            if( (image[n,m]) >= image_p100) and (image[n,m] <= image_p900):
                Np_10_90 += 1
                rMAD += np.abs(image[n,m])               
    rMAD /= Np_10_90
    '''
    print (N*M)
    print(image_p900)
    print( image_p100)
    print('\n\n', rMAD, Np_10_90)
    '''
    #----------------------------------------------------------------------       
    features = [
        image_min, 
        image_max, 
        image_mean, 
        image_std, 
        image_var,
        image_median,
        image_skew,
        image_kurtpsis,
        square_range,
        image_entropy,
        bins_entropy,
        image_norm_energy,
        image_rms,
        image_abs_deviation,
        image_p925,
        image_p850,
        image_p75,
        image_p150,
        interquartile_range,
        image_uniformity,
        rMAD]                   
    return features                        
#------------------------------------------------------------------------------ 
path = 'COVID-1.png'

#path = 'D:/Dropbox/Python_Imports/Data/Chest_X_Ray_collection_256x256_3_classes/COVID_resized/1.jpg'
image = skimage.io.imread(path)
Bins = 16 #Used to compute  the Bins entropy.
first_Oder_Statistics_features = Compute_First_Order_Statistics_Features(image, Bins)
#first_Oder_Statistics_features = np.round(first_Oder_Statistics_features, 3)
print('\n', first_Oder_Statistics_features)
#------------------------------------------------------------------------------    
    
    
    
    
    
    
    