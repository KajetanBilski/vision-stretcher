from typing import List, Tuple
import numpy as np
from seam_carving import seam_carve
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

UNSTRETCHABLE_CLASSES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'cow',
]

def get_classes():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

def init_detectron(unstretchables: List = None) -> Tuple[DefaultPredictor, set]:
    if unstretchables is None:
        unstretchables = UNSTRETCHABLE_CLASSES

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    classes_of_interest = []
    for i in unstretchables:
        classes_of_interest.append(metadata.thing_classes.index(i))
    
    return predictor, set(classes_of_interest)

def create_mask(predictor, masking_classes, img):
    outputs = predictor(img)
    masks = outputs['instances'].pred_masks.cpu().numpy()
    mask = np.zeros(img.shape[:2], dtype=bool)
    for i in range(len(outputs['instances'].pred_classes)):
        if outputs['instances'].pred_classes[i].item() in masking_classes:
            mask |= masks[i]
    return mask.astype(np.uint8) * 255

def resize(img, target_dims, mask = None):
    return seam_carve(img, target_dims[0] - img.shape[0], target_dims[1] - img.shape[1], mask)[0]

def pipeline(image, target_dims):
    predictor, coi = init_detectron()
    mask = create_mask(predictor, coi, image)
    output = resize(image, target_dims, mask)
    return output