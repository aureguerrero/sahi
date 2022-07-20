import os

from osgeo import gdal
from osgeo import osr
from osgeo import ogr

from sahi.utils.cv import read_image_as_pil
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy 
import struct
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import rotate
from scipy.signal import savgol_filter,find_peaks_cwt
import matplotlib.pyplot as ply
from skimage.morphology import skeletonize
import glob
from skimage import measure
import time

#====================================================================================

def calcularAnguloPromedio(lineas):
    
    height,width = lineas.shape
    lineas_label = measure.label(lineas, background=0, connectivity=2,return_num=False)
    cant = len(np.unique(lineas_label [int(height*0.1):-int(height*0.1),int(width*0.1):-int(width*0.1)]))-1
    angles = []
    for i in range(1,cant+1):
        y,x = np.where(lineas_label [int(height*0.1):-int(height*0.1),int(width*0.1):-int(width*0.1)]==i)
        if len(x)>0:
            left_x=np.min(x)
            left_x_pos=np.argmin(x)
            left_y=y[left_x_pos]
            rigth_x=np.max(x)
            rigth_x_pos=np.argmax(x)
            rigth_y=y[rigth_x_pos]
            angles.append( np.rad2deg( np.arctan2(rigth_y-left_y,rigth_x-left_x ) ) )
    angulo = np.mean(angles)

    return angulo

#====================================================================================

def recortoZonaSinCortarLineas(img_lines_aux_norm_rotada,proporcion):

    height,width = img_lines_aux_norm_rotada.shape

    y_crop_top = int(height*(proporcion/2))
    y_crop_bottom =np.min([ -y_crop_top,-1])
    x_crop_left = int(width*(proporcion/2))
    x_crop_rigth = np.min([-x_crop_left,-1])

    img_lines_aux = np.zeros_like(img_lines_aux_norm_rotada)
    img_lines_aux [ img_lines_aux_norm_rotada >= 0.2] = 0
    img_lines_aux [ img_lines_aux_norm_rotada < 0.2] = 1
    skele_new = skeletonize(img_lines_aux)
    transecta = skele_new[y_crop_top:y_crop_bottom,int(width*0.5)]
    entreLineas = np.where(transecta==1)

    y_crop_top_modified = y_crop_top+entreLineas[0][0]
    y_crop_bottom_modified = y_crop_top+entreLineas[0][-1]
    
    return y_crop_top_modified,y_crop_bottom_modified,x_crop_left,x_crop_rigth,entreLineas

#====================================================================================

def thresholdImageUsingSectors(image,block_size=512, thresholdType='percentile', value=97):
    ''' Apply threshold to an image by squared sectors

    Arguments:
    image -- 2D numpy array
    block_size -- Sector size in pixels, default is 512
    thresholdType -- Type of threshhold. 'fixed' or 'percentile'(default) 
    value -- threshold value in 'fixedValue', or prencitile limit in 'percetile'

    Return:
    uint8 2D numpy array with ones and zeros.

    '''
    
    height,width = image.shape

    nx = int(np.ceil(width/block_size))
    ny = int(np.ceil(height/block_size))

    thresh_image = np.zeros((height,width), dtype=np.uint8) 
    
    for x in range(nx):
        for y in range(ny):
            xi = x*block_size
            xf = ((x+1)*block_size)
            yi = y*block_size
            yf = ((y+1)*block_size)
            if (x==nx): xf=width-1
            if (y==ny): yf=height-1

            crop = np.asarray( image[yi:yf,xi:xf], dtype=float)        
 
            if  thresholdType=='fixedValue':
                thresh = value
            elif thresholdType=='percentile':
                thresh = np.nanpercentile(crop,value)

            thresh_image_aux = (crop*0).astype(dtype=np.uint8)  # np.zeros_like throws pylint error
            thresh_image_aux[crop>thresh] = 1
            thresh_image[yi:yf,xi:xf]=thresh_image_aux

    return thresh_image

#======================================================================================

def detectPlantsRGB(image,vari_percentile_threshold=94,a_channel_percentile_threshold=1):
    ''' Detect plants from a cv2 (OpenCv R-G-B ordered) image

    Arguments:
    image -- cv2 (OpenCv R-G-B ordered) image
    
    Return:
    uint8 2D numpy array with ones (plants) and zeros.

    '''
    
    # VARI    
    #-----------------------------------------------------------------------------
    
    r = image[:,:,2].astype('float')
    g = image[:,:,1].astype('float')
    b = image[:,:,0].astype('float')
    
    vari = (g-r)*100 / (g+r-b)
    min_vari = np.nanmin(vari)
    vari[np.isnan(vari)]=min_vari

    # RGB to LAB MODEL ( most negative a channel values are plants!)    
    #-----------------------------------------------------------------------------

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel,a_channel,b_channel = cv2.split(lab_image)


    # Threshold
    #-----------------------------------------------------------------------------
    #print(vari_percentile_threshold)
    #print(a_channel_percentile_threshold)
    vari_threshold = thresholdImageUsingSectors(vari,block_size=256, thresholdType='percentile', value=vari_percentile_threshold)
    a_channel_threshold = 1-thresholdImageUsingSectors(a_channel,block_size=256, thresholdType='percentile', value=a_channel_percentile_threshold)
    
    return a_channel_threshold,vari_threshold

#======================================================================================

def fftLines(image,fft_threshold=0.93):
    ''' Apply threshold to an image by squared sectors

    Arguments:
    image -- numpy 2D array
    fft_threshold=0.92 -- Threshold for the inverse FFT filter, relative to max. Default is 0.93
    

    Return:
    2D numpy array in range [-1,1].

    '''

    transf = np.fft.fft2(image-np.mean(image))
    transf_abs = np.abs(transf)
    transf_max = transf_abs.max()
    transf_abs[transf_abs<transf_max*fft_threshold]=0
    ifft = np.fft.ifft2(transf_abs*transf)
    ifft = (ifft / np.max(ifft))+1
    img_lines_aux = np.abs(ifft)

    # Hago umbral y Skeletonizo para separar las lineas de siembra    
    #-----------------------------------------------------------------------------
    return img_lines_aux/img_lines_aux.max()

#======================================================================================

def cal_resolucion(inputImage, d_surco_metros):
    image = np.array(read_image_as_pil(inputImage))
#     image = cv2.imread(inputImage)
    a_channel_threshold,vari_threshold = detectPlantsRGB(image)
    plantas_umbralizadas = a_channel_threshold*vari_threshold
    r = image[:,:,2].astype('float')
    g = image[:,:,1].astype('float')
    b = image[:,:,0].astype('float')
   
    # Calculo la FFT para direccion de linea de siembra    
    #-----------------------------------------------------------------------------
    img_lines_aux_norm = fftLines(plantas_umbralizadas)
    img_lines = np.zeros_like(img_lines_aux_norm)
    #img_lines [ img_lines_aux_norm >= 0.2] = 0
    img_lines [ img_lines_aux_norm < 0.2] = 1

    lineas_entre_siembra = skeletonize(img_lines)

    # Para cada linea, utilizo los extermos para calcular la pendiente     
    #-----------------------------------------------------------------------------
    angulo = calcularAnguloPromedio(lineas_entre_siembra) 

    # Roto y Recorto la imagen para quedarme con la parte central
    # y me fijo de no cortar la linea de siembra    
    #-----------------------------------------------------------------------------

    # Roto imgs
    img_lines_aux_norm_rotada = rotate(img_lines_aux_norm, angulo, reshape=False, mode='nearest')
    rgb = np.dstack((b,g,r)).astype('uint8') #For openCV
    #rgb = np.dstack((r,g,b)).astype('uint8')  #For Matplotlib
    rgb_rotada = rotate(rgb, angulo, reshape=False, mode='nearest')
    #vari_umbralizado_rotada = rotate(plantas_umbralizadas, angulo, reshape=False, mode='nearest')
    vari_umbralizado_rotada = rotate(vari_threshold, angulo, reshape=False, mode='nearest')
    a_thr_umbralizado_rotada = rotate(a_channel_threshold, angulo, reshape=False, mode='nearest')

    # Saco Zona de interés
    height,width = img_lines_aux_norm_rotada.shape
    proporcion=0.5
    y_crop_top = int(height*(proporcion/2))
    medio=int(np.median(np.arange(0,height)))
    img_lines_aux = np.zeros_like(img_lines_aux_norm_rotada)
    img_lines_aux [ img_lines_aux_norm_rotada >= 0.2] = 0
    img_lines_aux [ img_lines_aux_norm_rotada < 0.2] = 1
    skele_new = skeletonize(img_lines_aux)
    transecta = skele_new[y_crop_top:y_crop_bottom,int(width*0.5)]
    entreLineas = np.where(transecta==1)
    proporcion=2*np.min([np.max([entreLineas[np.max(np.where(entreLineas<medio)[0])]-10,0]),
                         np.max([height-entreLineas[np.min(np.where(entreLineas>medio)[0])]+10,0]),y_crop_top])/height
    y_crop_top,y_crop_bottom,x_crop_left,x_crop_rigth,entreLineas = recortoZonaSinCortarLineas(img_lines_aux_norm_rotada,proporcion)
    Nsurcos   = np.max([len(entreLineas[0]) -1,1])
    pix_surco = ( entreLineas[0][-1] - entreLineas[0][0] ) / Nsurcos
    return d_surco_metros / pix_surco  

