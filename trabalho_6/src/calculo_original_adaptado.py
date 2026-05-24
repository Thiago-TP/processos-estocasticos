# -*- coding: utf-8 -*-

import numpy as np
import scipy
import skimage
from skimage import feature, measure, data, color, exposure, io
import cv2

from collections import namedtuple

Features = namedtuple(
    "Features",
    [
        "minimo",
        "maximo",
        "media",
        "std",
        "var",
        "mediana",
        "assimetria",
        "curtose",
        "amplitude_ao_quadrado",
        "entropia_completa",
        "entropia_dos_bins",
        "energia_media",
        "rms",
        "desvio_absoluto_medio",
        "p925",
        "p850",
        "p150",
        "p75",
        "amplitude_interquartil",
        "uniformidade",
        "media_robusta",
    ],
)

def Compute_First_Order_Statistics_Features(image, Bins):        
    features = []   
    histRange = (0, 256) # the upper boundary is exclusive.
    image = image @ np.array([0.2125, 0.7154, 0.0721]) # Convertendo para escala de cinza
    N, M = image.shape  
    image = image.astype(np.uint8) # Convertendo para uint8 para garantir que os valores estejam na faixa de 0 a 255.
    image = image.flatten().astype(np.float64) # Achata a imagem e converte para float64 para os cálculos.
    # hist = cv2.calcHist(image, [0], None, [Bins], histRange)
    hist, _ = np.histogram(
        image,
        bins=Bins,
        range=histRange,
        density=False,
    )
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
    for i in range(len(image)):
        # for m in range(M):
        if( (image[i]) >= image_p100) and (image[i] <= image_p900):
            Np_10_90 += 1
            rMAD += np.abs(image[i]) # this is not a MAD...               
    rMAD /= Np_10_90
    '''
    print (N*M)
    print(image_p900)
    print( image_p100)
    print('\n\n', rMAD, Np_10_90)
    '''
    #----------------------------------------------------------------------       
    features = Features(
        minimo=image_min,
        maximo=image_max,
        media=image_mean,
        std=image_std,
        var=image_var,
        mediana=image_median,
        assimetria=image_skew,
        curtose=image_kurtpsis,
        amplitude_ao_quadrado=square_range,
        entropia_completa=image_entropy,
        entropia_dos_bins=bins_entropy,
        energia_media=image_norm_energy,
        rms=image_rms,
        desvio_absoluto_medio=image_abs_deviation,
        p925=image_p925,
        p850=image_p850,
        p150=image_p150,
        p75=image_p75,
        amplitude_interquartil=interquartile_range,
        uniformidade=image_uniformity,
        media_robusta=rMAD,
    )                  
    return features                        
#------------------------------------------------------------------------------ 
if __name__ == "__main__":
    """Sugestão de uso das funções neste script."""
    from glob import glob
    from os import path
    from pprint import pprint

    pasta = path.join("trabalho_6", "imagens")
    imagens = glob(path.join(pasta, "*.png")) + glob(path.join(pasta, "*.jpg"))

    for arquivo in imagens:
        print(f"Extraindo features de {arquivo}...")
        image = skimage.io.imread(arquivo)
        Bins = 16 #Used to compute  the Bins entropy.
        first_Oder_Statistics_features = Compute_First_Order_Statistics_Features(image, Bins)
        pprint(dict(first_Oder_Statistics_features._asdict()))
#------------------------------------------------------------------------------    
    
    
    
    
    
    
    