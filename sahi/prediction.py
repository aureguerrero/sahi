# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import copy
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from osgeo import gdal, ogr
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, RANSACRegressor

import os
import PIL
import cv2
import scipy.spatial
from skimage.morphology import skeletonize
from skimage import measure
from scipy.ndimage import rotate

from sahi.annotation import ObjectAnnotation
from sahi.utils.coco import CocoAnnotation, CocoPrediction
from sahi.utils.cv import read_image_as_pil, visualize_object_predictions
from sahi.utils.file import Path
from sahi.utils.resol_cal import cal_resolucion

def mascara(object_prediction_list):
  mask=np.zeros(object_prediction_list[0].mask.full_shape,dtype=bool)
  for objeto in object_prediction_list:
            mask1 = objeto.mask.bool_mask
            mask[objeto.bbox.to_voc_bbox()[1]:objeto.bbox.to_voc_bbox()[1]+np.shape(mask1)[0],
                     objeto.bbox.to_voc_bbox()[0]:objeto.bbox.to_voc_bbox()[0]+np.shape(mask1)[1]]=mask[objeto.bbox.to_voc_bbox()[1]:
                                                                                                        objeto.bbox.to_voc_bbox()[1]+np.shape(mask1)[0],
                                                                                                        objeto.bbox.to_voc_bbox()[0]:objeto.bbox.to_voc_bbox()[0]
                                                                                                        +np.shape(mask1)[1]]+mask1
  return mask*(objeto.category.id+1)

def centroide(mask,shift_amount=[0,0]):
    if mask is not None:
        c=np.array(np.where(skeletonize(mask))).transpose()
        aux=np.sum(scipy.spatial.distance_matrix(c,c,p=2),axis=1)
        minimo=np.min(aux)
        aux=c[np.where(aux==minimo)[0][0],:]
        c=aux
        return [c[1]+shift_amount[0],c[0]+shift_amount[1]]

    
def ecua_lineas_d_surco(lineas_entre_siembra,t=0.015):
  pto_ini=np.array(np.where(lineas_entre_siembra))
  lineas = cv2.HoughLines(lineas_entre_siembra[int(0.125*lineas_entre_siembra.shape[0]):
  -int(0.125*lineas_entre_siembra.shape[0]),int(0.125*lineas_entre_siembra.shape[1]):
  -int(0.125*lineas_entre_siembra.shape[0])], 1, np.pi/720, 100)
  angulo=np.mean(lineas.squeeze(),axis=0)[1]-np.pi/2
  if np.tan(angulo)<=0:
    x0, y0  = int(np.min(pto_ini[1,:])+t*lineas_entre_siembra.shape[1]),np.max(pto_ini[0,:])- int(t*(lineas_entre_siembra.shape[0]-1))
    y_x0=np.where(lineas_entre_siembra[:,x0]>0)[0]
    entrox=0
    entroy=0
    med_d_y_x0=np.mean([y_x0[i+1]-y_x0[i] for i in range(len(y_x0)-1)])
    if len(y_x0)>1:
      med_d_y_x0=np.mean([y_x0[i+1]-y_x0[i] for i in range(len(y_x0)-1)])
      entroy=1
      if y_x0[-1]>=y0:
        x_y0=np.where(lineas_entre_siembra[y0,x0+int((y0-y_x0[-1])/(np.tan(angulo)+1e-6))+2:]>0)[0]
        x_y0=x_y0+x0+int((y0-y_x0[-1])/(np.tan(angulo)+1e-6))+2
      else:
        x_y0=np.where(lineas_entre_siembra[y0,:]>0)[0]
    else:
      x_y0=np.where(lineas_entre_siembra[y0,:]>0)[0]
    if len(x_y0)>1:
      med_d_x_y0=np.mean([x_y0[i+1]-x_y0[i] for i in range(len(x_y0)-1)])
      entrox=1
    
    i=0
    
    ptos=[[0,0]]
    if entroy==1:
      ptos.append([y_x0[i],x0])
      while i<(len(y_x0)-1):
        j=i+1
        d=y_x0[j]-y_x0[i]
        while d<0.75*med_d_y_x0 and j<(len(y_x0)-1):
          j=j+1
          d=y_x0[j]-y_x0[i]
        i=j
        if d>=0.75*med_d_y_x0:
          ptos.append([y_x0[j],x0])
      if y_x0[-1]-ptos[-1][0]>0.75*med_d_y_x0:
          ptos.append([y_x0[-1],x0])

    i=0
    
    if entrox==1:#len(x_y0)>1:
      ptos.append([y0,x_y0[i]])
      while i<(len(x_y0)-1):
        j=i+1
        d=x_y0[j]-x_y0[i]
        while d<0.75*med_d_x_y0 and j<(len(x_y0)-1):
          j=j+1
          d=x_y0[j]-x_y0[i]
        i=j
        if d>=0.75*med_d_x_y0:
          ptos.append([y0,x_y0[j]])
      if x_y0[-1]-ptos[-1][1]>0.75*med_d_x_y0:
          ptos.append([y0,x_y0[-1]])
    ptos.append([lineas_entre_siembra.shape[0]-1,lineas_entre_siembra.shape[1]-1])


  else:
    x0, y0  = int(np.min(pto_ini[1,:])+t*(lineas_entre_siembra.shape[1]-1)), int(np.min(pto_ini[0,:])+t*(lineas_entre_siembra.shape[0]-1))
    x_y0=np.where(lineas_entre_siembra[y0,:]>0)[0][::-1]
    #med_d_x_y0=np.mean([x_y0[i]-x_y0[i+1] for i in range(len(x_y0)-1)])
    entrox=0
    entroy=0
    if len(x_y0)>1:
      med_d_x_y0=np.mean([x_y0[i]-x_y0[i+1] for i in range(len(x_y0)-1)])
      entrox=1
      if x_y0[-1]<x0:
        y_x0=np.where(lineas_entre_siembra[y0+int(np.tan(angulo))*(x0-x_y0[-1])+2:,x0]>0)[0]
        y_x0=y_x0+y0+np.tan(angulo)*(x0-x_y0[-1])+2
      else:
        y_x0=np.where(lineas_entre_siembra[:,x0]>0)[0]
    else:
      y_x0=np.where(lineas_entre_siembra[:,x0]>0)[0]
    if len(y_x0)>1:
      med_d_y_x0=np.mean([y_x0[i+1]-y_x0[i] for i in range(len(y_x0)-1)])
      entroy=1
    
    i=0
    
    ptos=[[0,lineas_entre_siembra.shape[1]-1]]
    if entrox==1:
      ptos.append([y0,x_y0[i]])
      while i<(len(x_y0)-1):
        j=i+1
        d=x_y0[i]-x_y0[j]
        while d<0.75*med_d_x_y0 and j<(len(x_y0)-1):
          j=j+1
          d=x_y0[i]-x_y0[j]
        i=j
        if d>=0.75*med_d_x_y0:
          ptos.append([y0,x_y0[j]])
        
      
      if ptos[-1][1]-x_y0[-1]>0.75*med_d_x_y0:
          ptos.append([y0,x_y0[-1]])

    
    i=0
    if entroy==1:
      ptos.append([y_x0[i],x0])
    
      while i<(len(y_x0)-1):
        j=i+1
        d=y_x0[j]-y_x0[i]
        while d<0.75*med_d_y_x0 and j<(len(y_x0)-1):
          j=j+1
          d=y_x0[j]-y_x0[i]
        i=j
        if d>=0.75*med_d_y_x0:
          ptos.append([y_x0[j],x0])
        
      if y_x0[-1]-ptos[-1][0]>0.75*med_d_y_x0:
          ptos.append([y_x0[-1],x0])
    ptos.append([lineas_entre_siembra.shape[0]-1,0])

  ptos_ini=np.array(ptos,dtype=np.int64)
  ptos=np.array(ptos,dtype=np.float64)
  ptos[:,1]=np.dot(ptos,np.array([[1.],[-np.tan(angulo)]]))[:,0]
  ptos[:,0]=np.tan(angulo)
  lineas=[np.poly1d(i) for i in ptos]
  return lineas,ptos_ini

#------------------------------------------------------

class PredictionScore:
    def __init__(self, value: float):
        """
        Arguments:
            score: prediction score between 0 and 1
        """
        # if score is a numpy object, convert it to python variable
        if type(value).__module__ == "numpy":
            value = copy.deepcopy(value).tolist()
        # set score
        self.value = value

    def is_greater_than_threshold(self, threshold):
        """
        Check if score is greater than threshold
        """
        return self.value > threshold

    def __repr__(self):
        return f"PredictionScore: <value: {self.value}>"


class ObjectPrediction(ObjectAnnotation):
    """
    Class for handling detection model predictions.
    """

    def __init__(
        self,
        bbox: Optional[List[int]] = None,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        bool_mask: Optional[np.ndarray] = None,
        score: Optional[float] = 0,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectPrediction from bbox, score, category_id, category_name, bool_mask.

        Arguments:
            bbox: list
                [minx, miny, maxx, maxy]
            score: float
                Prediction score between 0 and 1
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            bool_mask: np.ndarray
                2D boolean mask array. Should be None if model doesn't output segmentation mask.
            shift_amount: list
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        self.score = PredictionScore(score) 
  
       # self.bbox.to_voc_bbox()=bbox.to_voc_bbox()
        super().__init__(
            bbox=bbox,
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def centroide(self):
        if self.mask:
            return centroide(self.mask.bool_mask,[self.bbox.to_voc_bbox()[0],self.bbox.to_voc_bbox()[1]])
    
    def get_shifted_object_prediction(self):
        """
        Returns shifted version ObjectPrediction.
        Shifts bbox and mask coords.
        Used for mapping sliced predictions over full image.
        """
        if self.mask:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_voc_bbox(),
                category_id=self.category.id,
                score=self.score.value,
                bool_mask=self.mask.get_shifted_mask().bool_mask,
                category_name=self.category.name,
                shift_amount= self.mask.get_shifted_mask().shift_amount,#[0, 0],
                full_shape=self.mask.get_shifted_mask().full_shape,
            )
        else:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_voc_bbox(),
                category_id=self.category.id,
                score=self.score.value,
                bool_mask=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=None,
            )

    def to_coco_prediction(self, image_id=None):
        """
        Returns sahi.utils.coco.CocoPrediction representation of ObjectAnnotation.
        """
        if self.mask:
            coco_prediction = CocoPrediction.from_coco_segmentation(
                segmentation=self.mask.to_coco_segmentation(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.value,
                image_id=image_id,
            )
        else:
            coco_prediction = CocoPrediction.from_coco_bbox(
                bbox=self.bbox.to_coco_bbox(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.value,
                image_id=image_id,
            )
        return coco_prediction

    def to_fiftyone_detection(self, image_height: int, image_width: int):
        """
        Returns fiftyone.Detection representation of ObjectPrediction.
        """
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError('Please run "pip install -U fiftyone" to install fiftyone first for fiftyone conversion.')

        x1, y1, x2, y2 = self.bbox.to_voc_bbox()
        rel_box = [x1 / image_width, y1 / image_height, (x2 - x1) / image_width, (y2 - y1) / image_height]
        fiftyone_detection = fo.Detection(label=self.category.name, bounding_box=rel_box, confidence=self.score.value)
        return fiftyone_detection

    def __repr__(self):
        return f"""ObjectPrediction<
    bbox: {self.bbox},
    mask: {self.mask},
    score: {self.score},
    category: {self.category}>"""


class PredictionResult:
    def __init__(
        self,
        object_prediction_list: List[ObjectPrediction],
        image: Union[Image.Image, str, np.ndarray],
        durations_in_seconds: Optional[Dict] = None,
    ):
        self.image: Image.Image = read_image_as_pil(image)
        self.image_width, self.image_height = self.image.size
        self.object_prediction_list: List[ObjectPrediction] = object_prediction_list
        self.durations_in_seconds = durations_in_seconds
        self.centroides=[i.centroide() for i in object_prediction_list]
#         self.mascara=mascara(self.object_prediction_list)

       
       
    def clases(self):
        clases=[]
        for objeto in self.object_prediction_list:
            clases.append(objeto.category.id)
        return clases
    
    def mascaras(self):
#         mask=np.zeros((self.image_height,self.image_width),dtype=np.uint8)
#         for objeto in self.object_prediction_list:
#             mask1 = objeto.mask.bool_mask*1#(objeto.category.id+1)
#             mask[objeto.bbox.to_voc_bbox()[1]:objeto.bbox.to_voc_bbox()[1]+np.shape(mask1)[0],
#                      objeto.bbox.to_voc_bbox()[0]:objeto.bbox.to_voc_bbox()[0]+np.shape(mask1)[1]]=mask[objeto.bbox.to_voc_bbox()[1]:
#                                                                                                         objeto.bbox.to_voc_bbox()[1]+np.shape(mask1)[0],
#                                                                                                         objeto.bbox.to_voc_bbox()[0]:objeto.bbox.to_voc_bbox()[0]
#                                                                                                         +np.shape(mask1)[1]]+mask1
#             mask[np.where(mask>0)]=objeto.category.id+1
        return mascara(self.object_prediction_list)
    
    def lineas(self, fft_threshold=0.93,nminppl=10,clear =None, cota_transf=0.2,cambio=1):
        image=self.mascaras().copy()*1
        centros=np.array(self.centroides.copy())
        transf = np.fft.fft2(image-np.mean(image))
        transf_abs = np.abs(transf)
        transf_max = transf_abs.max()
        mascara=self.mascaras().copy()
        transf_abs[transf_abs<transf_max*fft_threshold]=0
        ifft = np.fft.ifft2(transf_abs*transf)
        ifft = (ifft / np.max(ifft))+1
        img_lines_aux = np.abs(ifft)
        img_lines_aux_norm=img_lines_aux/img_lines_aux.max()
        img_lines = np.zeros_like(img_lines_aux_norm)
        img_lines [ img_lines_aux_norm < cota_transf] = 1
        lineas_entre_siembra = skeletonize(img_lines)
        lineas_entre_siembra.dtype=np.uint8
        rectas,ptos_ini=ecua_lineas_d_surco(lineas_entre_siembra)
        dist_media_rec=np.mean([0.4*np.abs(rectas[i].coef[1]-rectas[i+1].coef[1])/np.mean([np.sqrt(rectas[r].coef[0]**2+1) for r in range(len(rectas))]) for i in range(len(rectas)-1)])
        lineas_d_surcos=[]
        object_prediction_list=[]
        centro2=[]
        id_surco=0
        info_d_surcos=[]
        ptos_a_sacar=[]
        for i in range(len(rectas)-1):
          ubic=np.where((centros[:,1]>rectas[i](centros[:,0]))*(centros[:,1]<rectas[i+1](centros[:,0]))==True)[0]

          if len(ubic)>=nminppl:
            id_surco=id_surco+1
            datos=centros[ubic,:]
            orden=[np.where(datos[:,0]==np.sort(datos[:,0])[i])[0][0] for i in range(len(ubic))]
            ubic=ubic[orden]
            datos=centros[ubic,:]
            if rectas[0].coef[0]<0:
              media_altura=np.mean(np.array([0.4*np.mean(np.array(self.object_prediction_list[l].mask.shape))/np.cos(np.arctan(rectas[0].coef[0])+np.pi/2) for l in ubic]))
            else:
              media_altura=np.mean(np.array([0.4*np.mean(np.array(self.object_prediction_list[l].mask.shape))/np.sin(np.arctan(rectas[0].coef[0])+np.pi/2) for l in ubic]))
            media_altura=np.min([media_altura,dist_media_rec])
            theilsen=RANSACRegressor(residual_threshold=media_altura).fit(np.expand_dims(datos[:,0],axis=1),datos[:,1])
            info_d_surcos.append({'entre_rec':[rectas[i],rectas[i+1]],'ubic': ubic,'ecuac':np.poly1d([theilsen.estimator_.coef_[0],theilsen.estimator_.intercept_]),'x_i_x_f':[np.min(centros[ubic,0]),np.max(centros[ubic,0])],'inliers': theilsen.inlier_mask_})
            for t in range(len(info_d_surcos[-1]['inliers'])):
              if ((info_d_surcos[-1]['inliers'][t]==False) or ((np.abs(info_d_surcos[-1]['ecuac'](datos[t,0])-datos[t,1])/np.sqrt(1+theilsen.estimator_.coef_[0]**2))>media_altura)) and (clear is not None):
                ptos_a_sacar.append(info_d_surcos[-1]['ubic'][t])
#             if clear is not None:
#               info_d_surcos[-1]['ubic']=[info_d_surcos[-1]['ubic'][u] for u in range(len(info_d_surcos[-1]['ubic'])) if info_d_surcos[-1]['inliers'][u]==True]
#               info_d_surcos[-1]['inliers']=[info_d_surcos[-1]['inliers'][u] for u in range(len(info_d_surcos[-1]['inliers'])) if info_d_surcos[-1]['inliers'][u]==True]
#               info_d_surcos[-1]['x_i_x_f']=[np.min(centros[info_d_surcos[-1]['ubic'],0]),np.max(centros[info_d_surcos[-1]['ubic'],0])]
          else:
              ptos_a_sacar.extend(ubic)
#             u=[p for p in range(len(self.object_prediction_list)) if p not in ubic]
#             self.centroides=[self.centroides[t] for t in u]
#             centros=centros[u]
#             self.object_prediction_list=[self.object_prediction_list[t] for t in u]
#         if clear is not None:
#           self.object_prediction_list=[self.object_prediction_list[i] for i in range(len(self.object_prediction_list)) if i not in set(ptos_a_sacar)]
#           self.centroides=[self.centroides[i] for i in range(len(self.centroides)) if i not in set(ptos_a_sacar)]
        if len(ptos_a_sacar)>0 and cambio==1:
            object_prediction_list=[self.object_prediction_list[i] for i in range(len(self.object_prediction_list)) if i not in set(ptos_a_sacar)]
            self.object_prediction_list=object_prediction_list
            centros=np.array([self.centroides[i] for i in range(len(self.centroides)) if i not in set(ptos_a_sacar)])
            self.centroides=centros.tolist()
          
            for r in range(len(info_d_surcos)):
              info_d_surcos[r]['ubic']=np.where((centros[:,1]>info_d_surcos[r]['entre_rec'][0](centros[:,0]))*(centros[:,1]<info_d_surcos[r]['entre_rec'][1](centros[:,0]))==True)[0]
              datos=centros[info_d_surcos[r]['ubic'],:]
              orden=[np.where(datos[:,0]==np.sort(datos[:,0])[i])[0][0] for i in range(len(info_d_surcos[r]['ubic']))]
              info_d_surcos[r]['ubic']=info_d_surcos[r]['ubic'][orden]
              datos=centros[info_d_surcos[r]['ubic'],:]
              if rectas[0].coef[0]<0:
                media_altura=np.mean(0.5*np.array([np.mean(np.array(self.object_prediction_list[l].mask.shape))/np.cos(np.arctan(rectas[0].coef[0])+np.pi/2) for l in info_d_surcos[r]['ubic']]))
              else:
                media_altura=np.mean(0.5*np.array([np.mean(np.array(self.object_prediction_list[l].mask.shape))/np.sin(np.arctan(rectas[0].coef[0])+np.pi/2) for l in info_d_surcos[r]['ubic']]))
              media_altura=np.min([media_altura,dist_media_rec])
              
              theilsen=RANSACRegressor(residual_threshold=media_altura).fit(np.expand_dims(datos[:,0],axis=1),datos[:,1])
              info_d_surcos[r]['inliers']=theilsen.inlier_mask_#np.array([info_d_surcos[r]['inliers'][l] for l in range(len(info_d_surcos[r]['ubic'])) if info_d_surcos[r]['ubic'][l] not in set(ptos_a_sacar)])
              info_d_surcos[r]['x_i_x_f']=[np.min(centros[info_d_surcos[r]['ubic'],0]),np.max(centros[info_d_surcos[r]['ubic'],0])]

#             info_d_surcos[r]['ubic']=np.where((centros[:,1]>info_d_surcos[r]['entre_rec'][0](centros[:,0]))*(centros[:,1]<info_d_surcos[r]['entre_rec'][1](centros[:,0]))==True)[0]
#             info_d_surcos[r]['inliers']=[info_d_surcos[r]['inliers'][l] for l in range(len(info_d_surcos[r]['ubic'])) if info_d_surcos[r]['ubic'][l] not in set(ptos_a_sacar)]
#             info_d_surcos[r]['x_i_x_f']=[np.min(centros[info_d_surcos[r]['ubic'],0]),np.max(centros[info_d_surcos[r]['ubic'],0])]
    
        return [i['ecuac'] for i in info_d_surcos],info_d_surcos
    
#     def lineas(self, fft_threshold=0.93,nminppl=10,clear =None):
#         image=self.mascaras().copy()*1
#         centros=np.array(self.centroides.copy())
#         transf = np.fft.fft2(image-np.mean(image))
#         transf_abs = np.abs(transf)
#         transf_max = transf_abs.max()
#         mascara=self.mascaras().copy()
#         transf_abs[transf_abs<transf_max*fft_threshold]=0
#         ifft = np.fft.ifft2(transf_abs*transf)
#         ifft = (ifft / np.max(ifft))+1
#         img_lines_aux = np.abs(ifft)
#         img_lines_aux_norm=img_lines_aux/img_lines_aux.max()
#         img_lines = np.zeros_like(img_lines_aux_norm)
#         img_lines [ img_lines_aux_norm < 0.2] = 1
#         lineas_entre_siembra = skeletonize(img_lines)
#         extrem_izq=np.percentile(np.where(lineas_entre_siembra==True)[1],5)
#         extrem_derec=np.percentile(np.where(lineas_entre_siembra==True)[1],95)
#         lineas2=np.array([np.where(lineas_entre_siembra[:,int(extrem_izq)]==True),np.where(lineas_entre_siembra[:,int(extrem_derec)]==True)]).squeeze()
#         rectas=[np.poly1d([(lineas2[1,i]-lineas2[0,i])/(extrem_derec-extrem_izq),-(lineas2[1,i]-lineas2[0,i])/(extrem_derec-extrem_izq)*extrem_izq+lineas2[0,i]]) for i in range(len(lineas2[0]))]
#         lineas_d_surcos=[]
#         object_prediction_list=[]
#         centro2=[]
    
#         if len(np.where((centros[:,1]<rectas[0](centros[:,0]))*(centros[:,1]>0)== True)[0])>1:
#             ubica=np.where((centros[:,1]<rectas[0](centros[:,0]))*(centros[:,1]>0)== True)[0]
#             if len(np.where((centros[:,1]<rectas[0](centros[:,0]))*(centros[:,1]>0)== True)[0])<nminppl:
#                 u=[p for p in range(len(self.object_prediction_list)) if p not in ubica]
#                 self.centroides=[self.centroides[t] for t in u]
#                 centros=centros[u]
#                 self.object_prediction_list=[self.object_prediction_list[t] for t in u]
#             else:
#                 datos=centros[np.where((centros[:,1]<rectas[0](centros[:,0]))*(centros[:,1]>0)== True),:].squeeze()
#     #             huber = HuberRegressor().fit(np.expand_dims(datos[:,0],axis=1),datos[:,1])
#     #             ubica2=np.where(np.abs(datos[:,1]-huber.predict(np.expand_dims(datos[:,0],axis=-1)))>0.5*np.mean(np.array([self.object_prediction_list[l].mask.shape[0] for l in ubica])))
#                 theilsen=TheilSenRegressor().fit(np.expand_dims(datos[:,0],axis=1),datos[:,1])
#                 ubica2=np.where(np.abs(datos[:,1]-theilsen.predict(np.expand_dims(datos[:,0],axis=-1)))>0.5*np.mean(np.array([self.object_prediction_list[l].mask.shape[0] for l in ubica])))


#                 if clear is not None:
#                     u=[p for p in range(len(self.object_prediction_list)) if p not in ubica[ubica2[0]]]
#                     self.centroides=[self.centroides[t] for t in u]
#                     centros=centros[u]
#                     self.object_prediction_list=[self.object_prediction_list[t] for t in u]
# #                     object_prediction_list.extend([self.object_prediction_list[t] for t in u])

#     #             lineas_d_surcos.append(np.poly1d([huber.coef_[0],huber.intercept_]))
#                 lineas_d_surcos.append(np.poly1d([theilsen.coef_[0],theilsen.intercept_]))
  
#         for i in range(len(rectas)-1):
#           if len(np.where((centros[:,1]<rectas[i+1](centros[:,0]))*(centros[:,1]>rectas[i](centros[:,0]))== True)[0])>1:
#             ubica=np.where((centros[:,1]<rectas[i+1](centros[:,0]))*(centros[:,1]>rectas[i](centros[:,0]))== True)[0]
#             if len(np.where((centros[:,1]<rectas[i+1](centros[:,0]))*(centros[:,1]>rectas[i](centros[:,0]))== True)[0])<nminppl:
#                 u=[p for p in range(len(self.object_prediction_list)) if p not in ubica]
#                 self.centroides=[self.centroides[t] for t in u]
#                 centros=centros[u]
#                 self.object_prediction_list=[self.object_prediction_list[t] for t in u]
#             else:
#                 datos=centros[np.where((centros[:,1]<rectas[i+1](centros[:,0]))*(centros[:,1]>rectas[i](centros[:,0]))== True),:].squeeze()
#     #             huber = HuberRegressor().fit(np.expand_dims(datos[:,0],axis=1),datos[:,1])
#     #             ubica2=np.where(np.abs(datos[:,1]-huber.predict(np.expand_dims(datos[:,0],axis=-1)))>0.5*np.mean(np.array([self.object_prediction_list[l].mask.shape[0] for l in ubica])))
#                 theilsen=TheilSenRegressor().fit(np.expand_dims(datos[:,0],axis=1),datos[:,1])
#                 ubica2=np.where(np.abs(datos[:,1]-theilsen.predict(np.expand_dims(datos[:,0],axis=-1)))>0.5*np.mean(np.array([self.object_prediction_list[l].mask.shape[0] for l in ubica])))


#                 if clear is not None:
#                     u=[p for p in range(len(self.object_prediction_list)) if p not in ubica[ubica2[0]]]
#                     self.centroides=[self.centroides[t] for t in u]
# #                     object_prediction_list=[self.object_prediction_list[t] for t in u]
#                     centros=centros[u]
#                     self.object_prediction_list=[self.object_prediction_list[t] for t in u]
                    

#     #             lineas_d_surcos.append(np.poly1d([huber.coef_[0],huber.intercept_]))
#                 lineas_d_surcos.append(np.poly1d([theilsen.coef_[0],theilsen.intercept_]))
    
#         if len(np.where((centros[:,1]>rectas[-1](centros[:,0]))*(centros[:,1]<mascara.shape[0])== True)[0])>1:
#             ubica=np.where((centros[:,1]>rectas[-1](centros[:,0]))*(centros[:,1]<mascara.shape[0])== True)[0]
#             if len(np.where((centros[:,1]>rectas[-1](centros[:,0]))*(centros[:,1]<mascara.shape[0])== True)[0])<nminppl:
#                 u=[p for p in range(len(self.object_prediction_list)) if p not in ubica]
#                 self.centroides=[self.centroides[t] for t in u]
#                 centros=centros[u]
#                 self.object_prediction_list=[self.object_prediction_list[t] for t in u]
#             else:
#                 datos=centros[np.where((centros[:,1]>rectas[-1](centros[:,0]))*(centros[:,1]<mascara.shape[0])== True),:].squeeze()
#     #             huber = HuberRegressor().fit(np.expand_dims(datos[:,0],axis=1),datos[:,1])
#     #             ubica2=np.where(np.abs(datos[:,1]-huber.predict(np.expand_dims(datos[:,0],axis=-1)))>0.5*np.mean(np.array([self.object_prediction_list[l].mask.shape[0] for l in ubica])))
#                 theilsen=TheilSenRegressor().fit(np.expand_dims(datos[:,0],axis=1),datos[:,1])
#                 ubica2=np.where(np.abs(datos[:,1]-theilsen.predict(np.expand_dims(datos[:,0],axis=-1)))>0.5*np.mean(np.array([self.object_prediction_list[l].mask.shape[0] for l in ubica])))


#                 if clear is not None:
#                     u=[p for p in range(len(self.object_prediction_list)) if p not in ubica[ubica2[0]]]
#                     self.centroides=[self.centroides[t] for t in u]
# #                     object_prediction_list=[self.object_prediction_list[t] for t in u]
#                     centros=centros[u]
#                     self.object_prediction_list=[self.object_prediction_list[t] for t in u]
                    

#     #             lineas_d_surcos.append(np.poly1d([huber.coef_[0],huber.intercept_]))
#                 lineas_d_surcos.append(np.poly1d([theilsen.coef_[0],theilsen.intercept_]))
        
#         #----------
         
#         id_surco=0
#         info_d_surcos=[]
#         if len(np.where((centros[:,1]<rectas[0](centros[:,0]))*(centros[:,1]>0)== True)[0])>nminppl:
#             ubica=np.where((centros[:,1]<rectas[0](centros[:,0]))*(centros[:,1]>0)== True)[0]
#             datos=centros[np.where((centros[:,1]<rectas[0](centros[:,0]))*(centros[:,1]>0)== True),:].squeeze()
#             orden=np.sort(datos[:,0])

#             info_d_surcos.append([id_surco,ubica[[np.where(datos[:,0]==np.sort(datos[:,0])[i])[0][0] for i in range(len(datos[:,0]))]]])
#             id_surco=id_surco+1
#             object_prediction_list.extend([self.object_prediction_list[u] for u in ubica])
#             centro2.extend([self.centroides[u] for u in ubica])
                                  
  
#         for i in range(len(rectas)-1):
#           if len(np.where((centros[:,1]<rectas[i+1](centros[:,0]))*(centros[:,1]>rectas[i](centros[:,0]))== True)[0])>nminppl:
#             ubica=np.where((centros[:,1]<rectas[i+1](centros[:,0]))*(centros[:,1]>rectas[i](centros[:,0]))== True)[0]
#             datos=centros[np.where((centros[:,1]<rectas[i+1](centros[:,0]))*(centros[:,1]>rectas[i](centros[:,0]))== True),:].squeeze()
#             orden=np.sort(datos[:,0])

#             info_d_surcos.append([id_surco,ubica[[np.where(datos[:,0]==np.sort(datos[:,0])[i])[0][0] for i in range(len(datos[:,0]))]]])
#             id_surco=id_surco+1
#             object_prediction_list.extend([self.object_prediction_list[u] for u in ubica])
#             centro2.extend([self.centroides[u] for u in ubica])
    
#         if len(np.where((centros[:,1]>rectas[-1](centros[:,0]))*(centros[:,1]<mascara.shape[0])== True)[0])>nminppl:
#           ubica=np.where((centros[:,1]>rectas[-1](centros[:,0]))*(centros[:,1]<mascara.shape[0])== True)[0]
#           datos=centros[np.where((centros[:,1]>rectas[-1](centros[:,0]))*(centros[:,1]<mascara.shape[0])== True),:].squeeze()
#           orden=np.sort(datos[:,0])

#           info_d_surcos.append([id_surco,ubica[[np.where(datos[:,0]==np.sort(datos[:,0])[i])[0][0] for i in range(len(datos[:,0]))]]])
#           id_surco=id_surco+1
#           object_prediction_list.extend([self.object_prediction_list[u] for u in ubica])
#           centro2.extend([self.centroides[u] for u in ubica])
        
#         self.object_prediction_list=object_prediction_list
#         self.centroides=centro2
                    
#         return lineas_d_surcos,info_d_surcos
    def info(self,proporcion=0.5,d_surco_metros=0.52):
        lineas,info_d_surcos=self.lineas()
        centros=self.centroides
        media_d_pendientes=np.mean([k['ecuac'][1] for k in info_d_surcos])
        dist_b=np.mean([np.abs(info_d_surcos[k]['ecuac'][0]-info_d_surcos[k+1]['ecuac'][0]) for k in range(len(info_d_surcos)-1)])
        dist_rec=np.mean([np.abs(info_d_surcos[k]['ecuac'][0]-info_d_surcos[k+1]['ecuac'][0])/np.sqrt(1+media_d_pendientes**2) for k in range(len(info_d_surcos)-1)])
#------------190722--------------------
#         d_surco_metros=d_surco_metros*dist_b/dist_rec
#         pix_surco=int(dist_b)
# ------------190722----------------y agregué linea siguiente
        resol=cal_resolucion(self.mascaras(),d_surco_metros)
        
        
#         comente1406
        rotacion=np.arctan(np.mean(np.array([lineas[i][1] for i in range(len(lineas))])))
#         siembra=np.zeros((self.image_height,self.image_width),np.uint8)
#         for i in range(len(lineas)):
#             cv2.line(siembra,(0,int(lineas[i](0))),(self.image_width-1,int(lineas[i](self.image_width-1))),(255,255,255),2)
#         siembra_rotada= rotate(siembra, rotacion*180/np.pi, reshape=False, mode='nearest')
#         height,width = siembra_rotada.shape

#         y_crop_top = int(height*(proporcion/2))
#         y_crop_bottom = -y_crop_top
#         x_crop_left = int(width*(proporcion/2))
#         x_crop_rigth = -x_crop_left
 
#         skele_new = skeletonize(siembra_rotada/255)
    
#         transecta = skele_new[y_crop_top:y_crop_bottom,int(width*0.5)]
#         entreLineas = np.where(transecta==1)
#         if len(entreLineas[0])<2:
#             transecta=skele_new[:,int(width*0.5)]
#             entreLineas = np.where(transecta==1)

#         y_crop_top_modified = y_crop_top+entreLineas[0][0]
#         y_crop_bottom_modified = y_crop_top+entreLineas[0][-1]
        
#         Nsurcos   = len(entreLineas[0])
#         if Nsurcos>1:
#             pix_surco = ( entreLineas[0][-1] - entreLineas[0][0] ) / (Nsurcos-1)
#         else:
#             pix_surco = entreLineas[0][0]
        
        #cambie 1 por 'ubic'
        resumen=[]
        id=1
        for p,t in zip(lineas,info_d_surcos):
          area=[self.object_prediction_list[l].to_shapely_annotation().area*(resol*100)**2 for l in t['ubic']]
          dist=[np.sqrt((p(centros[t['ubic'][l]][0])-p(centros[t['ubic'][l+1]][0]))**2
                                 +(centros[t['ubic'][l]][0]-centros[t['ubic'][l+1]][0])**2)*(resol*100) for l in range(len(t['ubic'])-1)]
          dist_real=[np.sqrt((centros[t['ubic'][l]][1]-centros[t['ubic'][l+1]][1])**2
                                 +(centros[t['ubic'][l]][0]-centros[t['ubic'][l+1]][0])**2)*(resol*100) for l in range(len(t['ubic'])-1)]
          resumen.append({'id':id,'recta': p,
                         'largo':np.sqrt((p(centros[t['ubic'][0]][0])-p(centros[t['ubic'][-1]][0]))**2
                                 +(centros[t['ubic'][0]][0]-centros[t['ubic'][-1]][0])**2)*(resol*100)
# ---------------------190722----------------------------------
#           area=[len(np.where(self.object_prediction_list[l].mask.bool_mask==True)[0])*(d_surco_metros*100/(pix_surco))**2 for l in t['ubic']]
#           dist=[np.sqrt((p(centros[t['ubic'][l]][0])-p(centros[t['ubic'][l+1]][0]))**2
#                                  +(centros[t['ubic'][l]][0]-centros[t['ubic'][l+1]][0])**2)*(d_surco_metros*100/(pix_surco)) for l in range(len(t['ubic'])-1)]
#           dist_real=[np.sqrt((centros[t['ubic'][l]][1]-centros[t['ubic'][l+1]][1])**2
#                                  +(centros[t['ubic'][l]][0]-centros[t['ubic'][l+1]][0])**2)*(d_surco_metros*100/(pix_surco)) for l in range(len(t['ubic'])-1)]
#           resumen.append({'id':id,'recta': p,
#                          'largo':np.sqrt((p(centros[t['ubic'][0]][0])-p(centros[t['ubic'][-1]][0]))**2
#                                  +(centros[t['ubic'][0]][0]-centros[t['ubic'][-1]][0])**2)*(d_surco_metros*100/(pix_surco))
                         ,'plantas':t['ubic'],
                         'area':area,
                         'stadist_area':{'cant_plt':len(t['ubic']),'min':np.min(np.array(area)), 'max':np.max(np.array(area)),
                                         'promedio':np.mean(np.array(area)),'desv_std':np.std(np.array(area)),
                                         'CV':np.std(np.array(area))/np.mean(np.array(area))},
                         'distancias':dist,
                         'stadist_dist':{'min':np.min(np.array(dist)), 'max':np.max(np.array(dist)),
                                         'promedio':np.mean(np.array(dist)),'desv_std':np.std(np.array(dist)),
                                         'CV':np.std(np.array(dist))/np.mean(np.array(dist))},
                         'distancias_real':dist_real,
                         'stadist_dist_real':{'min':np.min(np.array(dist_real)), 'max':np.max(np.array(dist_real)),
                                         'promedio':np.mean(np.array(dist_real)),'desv_std':np.std(np.array(dist_real)),
                                         'CV':np.std(np.array(dist_real))/np.mean(np.array(dist_real))}})
          id=id+1
            #---------------------------------------------
        resumen.append({'id':'total','plantas':[],'area':[],'distancias':[],'distancias_real':[],'stadist_area':{}, 'stadist_dist':{},'stadist_dist_real':{}})
        for i in range(len(lineas)):
            resumen[-1]['plantas'].extend(resumen[i]['plantas'])
            resumen[-1]['area'].extend(resumen[i]['area'])
            resumen[-1]['distancias'].extend(resumen[i]['distancias'])
            resumen[-1]['distancias_real'].extend(resumen[i]['distancias_real'])
            
#           resumen[-1]['plantas']=list(set().union(resumen[-1]['plantas'],resumen[i]['plantas']))
#           resumen[-1]['area']=list(set().union(resumen[-1]['area'],resumen[i]['area']))
#           resumen[-1]['distancias']=list(set().union(resumen[-1]['distancias'],resumen[i]['distancias']))
#           resumen[-1]['distancias_real']=list(set().union(resumen[-1]['distancias'],resumen[i]['distancias_real']))
        area=np.array(resumen[-1]['area'])
        dist=np.array(resumen[-1]['distancias'])
        dist_real=np.array(resumen[-1]['distancias_real'])
        resumen[-1]['stadist_area']={'cant_plt':len(resumen[-1]['plantas']),'min':np.min(np.array(area)),
                                    'max':np.max(np.array(area)),
                                    'promedio':np.mean(np.array(area)),'desv_std':np.std(np.array(area)),
                                    'CV':np.std(np.array(area))/np.mean(np.array(area))}

        resumen[-1]['stadist_dist']={'min':np.min(np.array(dist)), 'max':np.max(np.array(dist)),
                                          'promedio':np.mean(np.array(dist)),'desv_std':np.std(np.array(dist)),
                                          'CV':np.std(np.array(dist))/np.mean(np.array(dist))}
        
        resumen[-1]['stadist_dist_real']={'min':np.min(np.array(dist_real)), 'max':np.max(np.array(dist_real)),
                                          'promedio':np.mean(np.array(dist_real)),'desv_std':np.std(np.array(dist_real)),
                                          'CV':np.std(np.array(dist_real))/np.mean(np.array(dist_real))}
        
#         return {'rotacion': rotacion*180/np.pi,'resolucion_rotacion' : d_surco_metros/pix_surco,'resolucion_orig': d_surco_metros/pix_surco,'resumen':resumen}
        return {'rotacion': rotacion*180/np.pi,'resolucion_rotacion' : resol,'resolucion_orig': resol,'resumen':resumen}
           
    def export_visuals(self, export_dir: str = "demo_data/", export_file: str = "prediction_visual", text_size: float = None, text_th: float = None, rect_th: int = None, 
                       etiqueta: int =None, centro: int = None, lineas: int =None, clear=None, export_format: str = "png"):
        
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        image=np.array(self.image)
        mascara=self.mascaras()
        r = np.zeros_like(mascara).astype(np.uint8)
        g = np.zeros_like(mascara).astype(np.uint8)
        b = np.zeros_like(mascara).astype(np.uint8)
        colors = Colors()
        color = colors(self.object_prediction_list[0].category.id)
        (r[mascara > 0], g[mascara > 0], b[mascara >0]) = color
        
        rgb_mask = np.stack([r, g, b], axis=2)
        image = cv2.addWeighted(image, 1, rgb_mask, 0.4, 0)
        
        if centro is not None and centro !=0:
            centro=self.centroides
            ptos=np.zeros_like(image,dtype=np.uint8)
            centro=np.array(centro)
            ptos[centro[:,1],centro[:,0],:]=255
            kernel = np.ones((7,7),np.uint8)
            image = cv2.addWeighted(image, 1, cv2.dilate(ptos,kernel,iterations = 1), 0.8, 0)
            for i in centro:
                cv2.circle(image, i, 7, (255, 255, 255), -1)
#                 cv2.circle(image, i, 30, (255, 0, 0), 2)
        if lineas is not None and lineas !=0:
           lineas,info_d_surcos=self.lineas()
           for i in info_d_surcos:
                cv2.line(image,(i['x_i_x_f'][0],int(i['ecuac'](i['x_i_x_f'][0]))),(i['x_i_x_f'][1],int(i['ecuac'](i['x_i_x_f'][1]))),(255,255,255),3)
                
        if etiqueta is not None and etiqueta !=0:
            rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.001), 1)
            # set text_th for category names
            text_th = text_th or max(rect_th - 1, 1)
            # set text_size for category names
            text_size = text_size or rect_th / 3
            # add bbox and mask to image if present
            for object_prediction in self.object_prediction_list:
                bbox = object_prediction.bbox.to_voc_bbox()
                category_name = object_prediction.category.name
                score = object_prediction.score.value
                # set bbox points
                p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                # visualize boxes
                cv2.rectangle(
                    image,
                    p1,
                    p2,
                    color=color,
                    thickness=rect_th,
                )
                # arange bounding box text location
                label = f"{category_name} {score:.2f}"
                w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                # add bounding box text
                cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    image,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    text_size,
                    (255, 255, 255),
                    thickness=text_th,
                )
            
        if export_dir:
            # save inference result
            save_path = os.path.join(export_dir, export_file + "." + export_format)
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            
#         visualize_object_predictions(
#         image=np.ascontiguousarray(image),
#         etiqueta=etiqueta,
#         object_prediction_list=self.object_prediction_list,
#         rect_th=rect_th,
#         text_size=text_size,
#         text_th=None,
#         color=None,
#         output_dir=export_dir,
#         file_name=export_file,
#         export_format="png",
#         )
        
       
    def to_coco_annotations(self):
        coco_annotation_list = []
        for object_prediction in self.object_prediction_list:
            coco_annotation_list.append(object_prediction.to_coco_prediction().json)
        return coco_annotation_list

    def to_coco_predictions(self, image_id: Optional[int] = None):
        coco_prediction_list = []
        for object_prediction in self.object_prediction_list:
            coco_prediction_list.append(object_prediction.to_coco_prediction(image_id=image_id).json)
        return coco_prediction_list

    def to_imantics_annotations(self):
        imantics_annotation_list = []
        for object_prediction in self.object_prediction_list:
            imantics_annotation_list.append(object_prediction.to_imantics_annotation())
        return imantics_annotation_list

    def to_fiftyone_detections(self):
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError('Please run "pip install -U fiftyone" to install fiftyone first for fiftyone conversion.')

        fiftyone_detection_list: List[fo.Detection] = []
        for object_prediction in self.object_prediction_list:
            fiftyone_detection_list.append(
                object_prediction.to_fiftyone_detection(image_height=self.image_height, image_width=self.image_width)
            )
        return fiftyone_detection_list

   
class Colors:
    # color palette
    def __init__(self):
        hex = (
            "FF3838",
            "2C99A8",
            "FF701F",
            "6473FF",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "FF9D97",
            "00C2FF",
            "344593",
            "FFB21D",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

def select_random_color():
    """
    Selects random color.
    """
    colors = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
    ]
    return colors[random.randrange(0, 10)]
