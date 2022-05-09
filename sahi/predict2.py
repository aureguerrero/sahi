# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
import os
import time
import torch
import warnings
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
)
from sahi.postprocess.combine import batched_greedy_nmm, greedy_nmm
from sahi.annotation import BoundingBox
from sahi.postprocess.legacy.combine import UnionMergePostprocess
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.slicing import slice_image
from sahi.postprocess.utils import ObjectPredictionList, has_match, merge_object_prediction_pair
from sahi.utils.coco import Coco, CocoImage
from sahi.utils.cv import crop_object_predictions, read_image_as_pil, visualize_object_predictions
from sahi.utils.file import Path, import_class, increment_path, list_files, save_json, save_pickle
from skimage.morphology import skeletonize


MODEL_TYPE_TO_MODEL_CLASS_NAME = {
    "mmdet": "MmdetDetectionModel",
    "yolov5": "Yolov5DetectionModel",
    "detectron2": "Detectron2DetectionModel",
}

LOW_MODEL_CONFIDENCE = 0.1


logger = logging.getLogger(__name__)


def get_prediction(
    image,
    detection_model,
    image_size: int = None,
    shift_amount: list = [0, 0],
    full_shape=None,
    postprocess: Optional[PostprocessPredictions] = None,
    verbose: int = 0,
) -> PredictionResult:
    """
    Function for performing prediction for given image using given detection_model.

    Arguments:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionMode
        image_size: int
            Inference input size.
        shift_amount: List
            To shift the box and mask predictions from sliced image to full
            sized image, should be in the form of [shift_x, shift_y]
        full_shape: List
            Size of the full image, should be in the form of [height, width]
        postprocess: sahi.postprocess.combine.PostprocessPredictions
        verbose: int
            0: no print (default)
            1: print prediction duration

    Returns:
        A dict with fields:
            object_prediction_list: a list of ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    if image_size is not None:
        warnings.warn("Set 'image_size' at DetectionModel init.", DeprecationWarning)

    durations_in_seconds = dict()

    # read image as pil
    image_as_pil = read_image_as_pil(image)
    # get prediction
    time_start = time.time()
    detection_model.perform_inference(np.ascontiguousarray(image_as_pil), image_size=image_size)
    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    # process prediction
    time_start = time.time()
    # works only with 1 batch
#     detection_model.convert_original_predictions(
#         shift_amount=shift_amount,
#         full_shape=full_shape,
#     )
    
    object_prediction_list=[]
    for ten in range(len(detection_model.original_predictions['instances'])):
        boxes=list(detection_model.original_predictions['instances'][ten].get_fields()['pred_boxes'].tensor[0][:].int().numpy()+np.array(shift_amount+shift_amount))
        mask=detection_model.original_predictions['instances'][ten].get_fields()['pred_masks'].numpy().squeeze()
#         mask=mask[boxes[1]:boxes[3]+1,boxes[0]:boxes[2]+1]
        clase=int(detection_model.original_predictions['instances'][ten].get_fields()['pred_classes'])
        score=float(detection_model.original_predictions['instances'][ten].get_fields()['scores'])
        object_prediction_list.append(ObjectPrediction(
            bbox=boxes,bool_mask=mask,
            category_name=detection_model.category_names[clase],
            full_shape= full_shape,
            category_id=clase,score=score))
        object_prediction_list[ten].mask.bool_mask=object_prediction_list[ten].mask.bool_mask[
            object_prediction_list[ten].bbox.miny:object_prediction_list[ten].bbox.maxy+1,
            object_prediction_list[ten].bbox.minx:object_prediction_list[ten].bbox.maxx+1]
        object_prediction_list[ten].bbox=BoundingBox(list(np.array( 
            object_prediction_list[ten].bbox.to_voc_bbox())+np.array(shift_amount+shift_amount)))
    
#    object_prediction_list: List[ObjectPrediction] = detection_model.object_prediction_list
    
#     o=0
#     for object_prediction in detection_model.object_prediction_list:
#         if object_prediction:  # if not empty
#             x = object_prediction.bbox.to_coco_bbox()
#             object_prediction.bbox.shift_x=x[0]
#             object_prediction.bbox.shift_y=x[1]
#             object_prediction.mask.shift_x=x[0]
#             object_prediction.mask.shift_y=x[1]
#             object_prediction.mask.bool_mask=object_prediction.mask.bool_mask[x[1]:x[1]+x[3]+1,x[0]:x[0]+x[2]+1]
#             #object_prediction_list.append(object_prediction)
#             detection_model.object_prediction_list[o]=object_prediction
#             o=o+1
#     object_prediction_list: List[ObjectPrediction] = detection_model.object_prediction_list
#         del detection_model

    # postprocess matching predictions
    if postprocess is not None:
        object_prediction_list = postprocess(object_prediction_list)

    time_end = time.time() - time_start
    durations_in_seconds["postprocess"] = time_end

    if verbose == 1:
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    return PredictionResult(
        image=image, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds
    )


def get_sliced_prediction(
    image,
    detection_model=None,
    image_size: int = None,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    perform_standard_pred: bool = True,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
) -> PredictionResult:
    """
    Function for slice image + get predicion for each slice + combine predictions in full image.

    Args:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionModel
        image_size: int
            Input image size for each inference (image is scaled by preserving asp. rat.).
        slice_height: int
            Height of each slice.  Defaults to ``512``.
        slice_width: int
            Width of each slice.  Defaults to ``512``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        perform_standard_pred: bool
            Perform a standard prediction on top of sliced predictions to increase large object
            detection accuracy. Default: True.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        verbose: int
            0: no print
            1: print number of slices (default)
            2: print number of slices and slice/prediction durations

    Returns:
        A Dict with fields:
            object_prediction_list: a list of sahi.prediction.ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    if image_size is not None:
        warnings.warn("Set 'image_size' at DetectionModel init.", DeprecationWarning)

    # for profiling
    durations_in_seconds = dict()

    # currently only 1 batch supported
    num_batch = 1

    # create slices from full image
    time_start = time.time()
    # slice_image_result = slice_image(
    #     image=image,
    #     slice_height=slice_height,
    #     slice_width=slice_width,
    #     overlap_height_ratio=overlap_height_ratio,
    #     overlap_width_ratio=overlap_width_ratio,
    # )
    # num_slices = len(slice_image_result)
    #------------------------------------------incorporado------------------------
    slice_image_result = slice_image(
        image=image,
        slice_height=2*slice_height,
        slice_width=2*slice_width,
        overlap_height_ratio=0,
        overlap_width_ratio=0,
    )
    #-----------------------------------------------------------------------------
    num_slices = len(slice_image_result)
    
    
    time_end = time.time() - time_start
    durations_in_seconds["slice"] = time_end

    # init match postprocess instance
    if postprocess_type in ["NMM"]:
        postprocess = NMMPostprocess(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )
    elif postprocess_type == "GREEDYNMM":
        postprocess = GreedyNMMPostprocess(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )
    elif postprocess_type == "NMS":
        postprocess = NMSPostprocess(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )
    elif postprocess_type == "LSNMS":
        postprocess = LSNMSPostprocess(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )
    elif postprocess_type == "UNIONMERGE":
        # sahi v0.8.16 compatibility
        logger.warning("'UNIONMERGE' is deprecated, use 'GREEDYNMM' instead.")
        postprocess = UnionMergePostprocess(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )
    else:
        raise ValueError(
            f"postprocess_type should be one of ['NMS', 'NMM', 'GREEDYNMM'] but given as {postprocess_type}"
        )

    # create prediction input
    num_group = int(num_slices / num_batch)
    if verbose == 1 or verbose == 2:
        tqdm.write(f"Performing prediction on {num_slices} number of slices.")
    object_prediction_list = []
    # perform sliced prediction
    for group_ind in range(num_group):
        # prepare batch (currently supports only 1 batch)
        image_list = []
        shift_amount_list = []
        for image_ind in range(num_batch):
            image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
            shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])
        # perform batch prediction
        prediction_result = get_prediction(
            image=image_list[0],
            detection_model=detection_model,
            image_size=image_size,
            shift_amount=shift_amount_list[0],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
        )
        object_prediction_list.extend(prediction_result.object_prediction_list)
    mask=np.zeros((object_prediction_list[0].mask.full_shape_height,
                   object_prediction_list[0].mask.full_shape_width),dtype=np.uint8)
    for objeto in object_prediction_list:
        mask1 = objeto.mask.bool_mask*1
        mask[objeto.bbox.to_voc_bbox()[1]:objeto.bbox.to_voc_bbox()[1]+np.shape(mask1)[0],
             objeto.bbox.to_voc_bbox()[0]:objeto.bbox.to_voc_bbox()[0]+np.shape(mask1)[1]]= mask[objeto.bbox.to_voc_bbox()[1]:
                                                                                                objeto.bbox.to_voc_bbox()[1]+np.shape(mask1)[0],
                                                                                                objeto.bbox.to_voc_bbox()[0]:
                                                                                                objeto.bbox.to_voc_bbox()[0]+np.shape(mask1)[1]]+mask1
    mask[np.where(mask)>1]=objeto.category.id+1
    
    transf = np.fft.fft2(mask-np.mean(mask))
    transf_abs = np.abs(transf)
    transf_max = transf_abs.max()
    transf_abs[transf_abs<transf_max*0.93]=0
    ifft = np.fft.ifft2(transf_abs*transf)
    ifft = (ifft / np.max(ifft))+1
    img_lines_aux = np.abs(ifft)
    img_lines_aux_norm=img_lines_aux/img_lines_aux.max()
    img_lines = np.zeros_like(img_lines_aux_norm)
    img_lines [ img_lines_aux_norm < 0.2] = 1
    lineas_entre_siembra = skeletonize(img_lines)
    extrem_izq=np.percentile(np.where(lineas_entre_siembra==True)[1],5)
    extrem_derec=np.percentile(np.where(lineas_entre_siembra==True)[1],95)
    lineas2=np.array([np.where(lineas_entre_siembra[:,int(extrem_izq)]==True),np.where(lineas_entre_siembra[:,int(extrem_derec)]==True)]).squeeze()
    lineas2=list(lineas2)
    if lineas2[-1]<np.shape(mask)[0]:
      lineas2.append(np.shape(mask)[0])
    if lineas2[0]>0:
      lineas2.insert(0,0)
    
    limites=[[lineas2[i],lineas2[i+1]] for i in range(len(lineas2)-1)]
    ima=read_image_as_pil(image)
    imag_surc=[np.array(ima)[i[0]:i[1],:,:] for i in limites]
    object_prediction_list=[]
    for l in range(len(imag_surc)):
        slice_s=slice_image(imag_surc[l],slice_height=
                            int(np.max([slice_height,limites[l][1]-limites[0][1]+1])),slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,overlap_width_ratio=overlap_width_ratio,)
        for t in range(len(slice_s.starting_pixels)):
            slice_s.starting_pixels[t][1]=limites[l][0]
            prediction_result = get_prediction(
                image=slice_s.images[t],
                detection_model=detection_model,
                image_size=image_size,
                shift_amount=slice_s.starting_pixels[t],
                full_shape=[
                    slice_image_result.original_image_height,
                    slice_image_result.original_image_width,
                ],
            )
            object_prediction_list.extend(prediction_result.object_prediction_list)

    print(len(object_prediction_list))
                
    # perform standard prediction
    if num_slices > 1 and perform_standard_pred:
        prediction_result = get_prediction(
            image=image,
            detection_model=detection_model,
            image_size=image_size,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None,
        )
        tam=len(prediction_result.object_prediction_list.copy())
        prediction_result.object_prediction_list.extend(object_prediction_list)
        agregar_t= batched_greedy_nmm(ObjectPredictionList(prediction_result.object_prediction_list).totensor(),
                              match_threshold=0.01,
                              match_metric=postprocess_match_metric,
                              )
        agrega_t=list(agregar_t.items())
        tam2=len(prediction_result.object_prediction_list)
#         prediction_result.object_prediction_list=[prediction_result.object_prediction_list[agrega_t[i][0]] for i in range(len(agrega_t)) if agrega_t[i][0]<tam and len(set(agrega_t[i][1]) & set(list(range(tam,tam2))))==0]
        conj=set()
        for i in range(len(agrega_t)):
            if agrega_t[i][0]<tam and len(set(agrega_t[i][1]) & set(list(range(tam,tam2))))>0:
                conj=set.union(conj,set([agrega_t[i][0]]))
            if  agrega_t[i][0]>=tam and len(set(agrega_t[i][1]) & set(list(range(tam))))>0:
                conj=set.union(conj,set(agrega_t[i][1]) & set(list(range(tam))))
        prediction_result.object_prediction_list=[prediction_result.object_prediction_list[i] for i in list(set(list(range(tam)))-conj)]
    object_prediction_list.extend(prediction_result.object_prediction_list)
        #object_prediction_list.extend(prediction_agregar2)
    del prediction_result,agregar_t

    print(len(object_prediction_list))
        
    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    if verbose == 2:
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    # merge matching predictions
    if len(object_prediction_list) > 0:
        object_prediction_list2=[]
        cant_old=(object_prediction_list2)
        cant_new=len(object_prediction_list)
        i=0
        while cant_old!=cant_new and i<=2:
            cant_old=len(object_prediction_list2)
            object_prediction_list2 = postprocess(object_prediction_list)
            ##if perform_standard_pred:
            ##    postprocess.match_threshold=postprocess.match_threshold+min([max([0,(1-postprocess.match_threshold)/2]),0.1])
            postprocess.match_metric="IOU"
            cant_new=len(object_prediction_list2)
            object_prediction_list=object_prediction_list2.copy()
            print(i,cant_old,cant_new)
            i=i+1
            
    print(len(object_prediction_list2))

    return PredictionResult(
        image=image, object_prediction_list=object_prediction_list2, durations_in_seconds=durations_in_seconds
    )


def predict(
    model_type: str = "mmdet",
    model_path: str = None,
    model_config_path: str = None,
    model_confidence_threshold: float = 0.25,
    model_device: str = None,
    model_category_mapping: dict = None,
    model_category_remapping: dict = None,
    source: str = None,
    no_standard_prediction: bool = False,
    no_sliced_prediction: bool = False,
    image_size: int = None,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    export_visual: bool = False,
    export_pickle: bool = False,
    export_crop: bool = False,
    dataset_json_path: bool = None,
    project: str = "runs/predict",
    name: str = "exp",
    visual_bbox_thickness: int = None,
    visual_text_size: float = None,
    visual_text_thickness: int = None,
    visual_export_format: str = "png",
    verbose: int = 1,
    return_dict: bool = False,
    force_postprocess_type: bool = False,
):
    """
    Performs prediction for all present images in given folder.

    Args:
        model_type: str
            mmdet for 'MmdetDetectionModel', 'yolov5' for 'Yolov5DetectionModel'.
        model_path: str
            Path for the model weight
        model_config_path: str
            Path for the detection model config file
        model_confidence_threshold: float
            All predictions with score < model_confidence_threshold will be discarded.
        model_device: str
            Torch device, "cpu" or "cuda"
        model_category_mapping: dict
            Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
        model_category_remapping: dict: str to int
            Remap category ids after performing inference
        source: str
            Folder directory that contains images or path of the image to be predicted.
        no_standard_prediction: bool
            Dont perform standard prediction. Default: False.
        no_sliced_prediction: bool
            Dont perform sliced prediction. Default: False.
        image_size: int
            Input image size for each inference (image is scaled by preserving asp. rat.).
        slice_height: int
            Height of each slice.  Defaults to ``512``.
        slice_width: int
            Width of each slice.  Defaults to ``512``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        export_pickle: bool
            Export predictions as .pickle
        export_crop: bool
            Export predictions as cropped images.
        dataset_json_path: str
            If coco file path is provided, detection results will be exported in coco json format.
        project: str
            Save results to project/name.
        name: str
            Save results to project/name.
        visual_bbox_thickness: int
        visual_text_size: float
        visual_text_thickness: int
        visual_export_format: str
            Can be specified as 'jpg' or 'png'
        verbose: int
            0: no print
            1: print slice/prediction durations, number of slices
            2: print model loading/file exporting durations
        return_dict: bool
            If True, returns a dict with 'export_dir' field.
        force_postprocess_type: bool
            If True, auto postprocess check will e disabled
    """
    # assert prediction type
    assert (
        no_standard_prediction and no_sliced_prediction
    ) is not True, "'no_standard_prediction' and 'no_sliced_prediction' cannot be True at the same time."

    # auto postprocess type
    if not force_postprocess_type and model_confidence_threshold < LOW_MODEL_CONFIDENCE and postprocess_type != "NMS":
        logger.warning(
            f"Switching postprocess type/metric to NMS/IOU since confidence threshold is low ({model_confidence_threshold})."
        )
        postprocess_type = "NMS"
        postprocess_match_metric = "IOU"

    # for profiling
    durations_in_seconds = dict()

    # list image files in directory
    if dataset_json_path:
        coco: Coco = Coco.from_coco_dict_or_path(dataset_json_path)
        image_path_list = [str(Path(source) / Path(coco_image.file_name)) for coco_image in coco.images]
        coco_json = []
    elif os.path.isdir(source):
        image_path_list = list_files(
            directory=source,
            contains=[".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
            verbose=verbose,
        )
    else:
        image_path_list = [source]

    # init export directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run
    crop_dir = save_dir / "crops"
    visual_dir = save_dir / "visuals"
    visual_with_gt_dir = save_dir / "visuals_with_gt"
    pickle_dir = save_dir / "pickles"
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # init model instance
    time_start = time.time()
    model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]
    DetectionModel = import_class(model_class_name)
    detection_model = DetectionModel(
        model_path=model_path,
        config_path=model_config_path,
        confidence_threshold=model_confidence_threshold,
        device=model_device,
        category_mapping=model_category_mapping,
        category_remapping=model_category_remapping,
        load_at_init=False,
        image_size=image_size,
    )
    detection_model.load_model()
    time_end = time.time() - time_start
    durations_in_seconds["model_load"] = time_end

    # iterate over source images
    durations_in_seconds["prediction"] = 0
    durations_in_seconds["slice"] = 0
    for ind, image_path in enumerate(tqdm(image_path_list, "Performing inference on images")):
        # get filename
        if os.path.isdir(source):  # preserve source folder structure in export
            relative_filepath = str(Path(image_path)).split(str(Path(source)))[-1]
            relative_filepath = relative_filepath[1:] if relative_filepath[0] == os.sep else relative_filepath
        else:  # no process if source is single file
            relative_filepath = Path(image_path).name
        filename_without_extension = Path(relative_filepath).stem
        # load image
        image_as_pil = read_image_as_pil(image_path)

        # perform prediction
        if not no_sliced_prediction:
            # get sliced prediction
            prediction_result = get_sliced_prediction(
                image=image_path,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                perform_standard_pred=not no_standard_prediction,
                postprocess_type=postprocess_type,
                postprocess_match_metric=postprocess_match_metric,
                postprocess_match_threshold=postprocess_match_threshold,
                postprocess_class_agnostic=postprocess_class_agnostic,
                verbose=1 if verbose else 0,
            )
            object_prediction_list = prediction_result.object_prediction_list
            durations_in_seconds["slice"] += prediction_result.durations_in_seconds["slice"]
        else:
            # get standard prediction
            prediction_result = get_prediction(
                image=image_path,
                detection_model=detection_model,
                shift_amount=[0, 0],
                full_shape=None,
                postprocess=None,
                verbose=0,
            )
            object_prediction_list = prediction_result.object_prediction_list

        durations_in_seconds["prediction"] += prediction_result.durations_in_seconds["prediction"]

        if dataset_json_path:
            # append predictions in coco format
            for object_prediction in object_prediction_list:
                coco_prediction = object_prediction.to_coco_prediction()
                coco_prediction.image_id = coco.images[ind].id
                coco_prediction_json = coco_prediction.json
                if coco_prediction_json["bbox"]:
                    coco_json.append(coco_prediction_json)
            if export_visual:
                # convert ground truth annotations to object_prediction_list
                coco_image: CocoImage = coco.images[ind]
                object_prediction_gt_list: List[ObjectPrediction] = []
                for coco_annotation in coco_image.annotations:
                    coco_annotation_dict = coco_annotation.json
                    category_name = coco_annotation.category_name
                    full_shape = [coco_image.height, coco_image.width]
                    object_prediction_gt = ObjectPrediction.from_coco_annotation_dict(
                        annotation_dict=coco_annotation_dict, category_name=category_name, full_shape=full_shape
                    )
                    object_prediction_gt_list.append(object_prediction_gt)
                # export visualizations with ground truths
                output_dir = str(visual_with_gt_dir / Path(relative_filepath).parent)
                color = (0, 255, 0)  # original annotations in green
                result = visualize_object_predictions(
                    np.ascontiguousarray(image_as_pil),
                    object_prediction_list=object_prediction_gt_list,
                    rect_th=visual_bbox_thickness,
                    text_size=visual_text_size,
                    text_th=visual_text_thickness,
                    color=color,
                    output_dir=None,
                    file_name=None,
                    export_format=None,
                )
                color = (255, 0, 0)  # model predictions in red
                _ = visualize_object_predictions(
                    result["image"],
                    object_prediction_list=object_prediction_list,
                    rect_th=visual_bbox_thickness,
                    text_size=visual_text_size,
                    text_th=visual_text_thickness,
                    color=color,
                    output_dir=output_dir,
                    file_name=filename_without_extension,
                    export_format=visual_export_format,
                )

        time_start = time.time()
        # export prediction boxes
        if export_crop:
            output_dir = str(crop_dir / Path(relative_filepath).parent)
            crop_object_predictions(
                image=np.ascontiguousarray(image_as_pil),
                object_prediction_list=object_prediction_list,
                output_dir=output_dir,
                file_name=filename_without_extension,
                export_format=visual_export_format,
            )
        # export prediction list as pickle
        if export_pickle:
            save_path = str(pickle_dir / Path(relative_filepath).parent / (filename_without_extension + ".pickle"))
            save_pickle(data=object_prediction_list, save_path=save_path)
        # export visualization
        if export_visual:
            output_dir = str(visual_dir / Path(relative_filepath).parent)
            visualize_object_predictions(
                np.ascontiguousarray(image_as_pil),
                object_prediction_list=object_prediction_list,
                rect_th=visual_bbox_thickness,
                text_size=visual_text_size,
                text_th=visual_text_thickness,
                output_dir=output_dir,
                file_name=filename_without_extension,
                export_format=visual_export_format,
            )
        time_end = time.time() - time_start
        durations_in_seconds["export_files"] = time_end

    # export coco results
    if dataset_json_path:
        save_path = str(save_dir / "result.json")
        save_json(coco_json, save_path)

    if export_visual or export_pickle or export_crop or dataset_json_path is not None:
        print(f"Prediction results are successfully exported to {save_dir}")

    # print prediction duration
    if verbose == 2:
        print(
            "Model loaded in",
            durations_in_seconds["model_load"],
            "seconds.",
        )
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )
        if export_visual:
            print(
                "Exporting performed in",
                durations_in_seconds["export_files"],
                "seconds.",
            )

    if return_dict:
        return {"export_dir": save_dir}


def predict_fiftyone(
    model_type: str = "mmdet",
    model_path: str = None,
    model_config_path: str = None,
    model_confidence_threshold: float = 0.25,
    model_device: str = None,
    model_category_mapping: dict = None,
    model_category_remapping: dict = None,
    dataset_json_path: str = None,
    image_dir: str = None,
    no_standard_prediction: bool = False,
    no_sliced_prediction: bool = False,
    image_size: int = None,
    slice_height: int = 256,
    slice_width: int = 256,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
):
    """
    Performs prediction for all present images in given folder.

    Args:
        model_type: str
            mmdet for 'MmdetDetectionModel', 'yolov5' for 'Yolov5DetectionModel'.
        model_path: str
            Path for the model weight
        model_config_path: str
            Path for the detection model config file
        model_confidence_threshold: float
            All predictions with score < model_confidence_threshold will be discarded.
        model_device: str
            Torch device, "cpu" or "cuda"
        model_category_mapping: dict
            Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
        model_category_remapping: dict: str to int
            Remap category ids after performing inference
        dataset_json_path: str
            If coco file path is provided, detection results will be exported in coco json format.
        image_dir: str
            Folder directory that contains images or path of the image to be predicted.
        no_standard_prediction: bool
            Dont perform standard prediction. Default: False.
        no_sliced_prediction: bool
            Dont perform sliced prediction. Default: False.
        image_size: int
            Input image size for each inference (image is scaled by preserving asp. rat.).
        slice_height: int
            Height of each slice.  Defaults to ``256``.
        slice_width: int
            Width of each slice.  Defaults to ``256``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        verbose: int
            0: no print
            1: print slice/prediction durations, number of slices, model loading/file exporting durations
    """
    from sahi.utils.fiftyone import create_fiftyone_dataset_from_coco_file, fo

    # assert prediction type
    assert (
        no_standard_prediction and no_sliced_prediction
    ) is not True, "'no_standard_pred' and 'no_sliced_prediction' cannot be True at the same time."

    # for profiling
    durations_in_seconds = dict()

    dataset = create_fiftyone_dataset_from_coco_file(image_dir, dataset_json_path)

    # init model instance
    time_start = time.time()
    model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]
    DetectionModel = import_class(model_class_name)
    detection_model = DetectionModel(
        model_path=model_path,
        config_path=model_config_path,
        confidence_threshold=model_confidence_threshold,
        device=model_device,
        category_mapping=model_category_mapping,
        category_remapping=model_category_remapping,
        load_at_init=False,
        image_size=image_size,
    )
    detection_model.load_model()
    time_end = time.time() - time_start
    durations_in_seconds["model_load"] = time_end

    # iterate over source images
    durations_in_seconds["prediction"] = 0
    durations_in_seconds["slice"] = 0
    # Add predictions to samples
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            # perform prediction
            if not no_sliced_prediction:
                # get sliced prediction
                prediction_result = get_sliced_prediction(
                    image=sample.filepath,
                    detection_model=detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    perform_standard_pred=not no_standard_prediction,
                    postprocess_type=postprocess_type,
                    postprocess_match_threshold=postprocess_match_threshold,
                    postprocess_match_metric=postprocess_match_metric,
                    postprocess_class_agnostic=postprocess_class_agnostic,
                    verbose=verbose,
                )
                durations_in_seconds["slice"] += prediction_result.durations_in_seconds["slice"]
            else:
                # get standard prediction
                prediction_result = get_prediction(
                    image=sample.filepath,
                    detection_model=detection_model,
                    shift_amount=[0, 0],
                    full_shape=None,
                    postprocess=None,
                    verbose=0,
                )
                durations_in_seconds["prediction"] += prediction_result.durations_in_seconds["prediction"]

            # Save predictions to dataset
            sample[model_type] = fo.Detections(detections=prediction_result.to_fiftyone_detections())
            sample.save()

    # print prediction duration
    if verbose == 1:
        print(
            "Model loaded in",
            durations_in_seconds["model_load"],
            "seconds.",
        )
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    # visualize results
    session = fo.launch_app()
    session.dataset = dataset
    # Evaluate the predictions
    results = dataset.evaluate_detections(
        model_type,
        gt_field="ground_truth",
        eval_key="eval",
        iou=postprocess_match_threshold,
        compute_mAP=True,
    )
    # Get the 10 most common classes in the dataset
    counts = dataset.count_values("ground_truth.detections.label")
    classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]
    # Print a classification report for the top-10 classes
    results.print_report(classes=classes_top10)
    # Load the view on which we ran the `eval` evaluation
    eval_view = dataset.load_evaluation_view("eval")
    # Show samples with most false positives
    session.view = eval_view.sort_by("eval_fp", reverse=True)
    while 1:
        time.sleep(3)
