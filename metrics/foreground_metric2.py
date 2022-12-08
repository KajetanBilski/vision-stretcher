import cv2
import numpy as np
from sklearn.metrics import jaccard_score

def crop_image(img):
    minx, miny, maxx, maxy = 0, 0, img.shape[1], img.shape[0]
    while not img[miny, :].any():
        miny += 1
    while not img[maxy - 1, :].any():
        maxy -= 1
    while not img[:, minx].any():
        minx += 1
    while not img[:, maxx - 1].any():
        maxx -= 1
    return img[miny:maxy, minx:maxx]

def foreground_score(src_img, rec_img, predictor, coi):
    src_outs, rec_outs = predictor(src_img), predictor(rec_img)
    src_masks, rec_masks = src_outs['instances'].pred_masks.cpu().numpy(), rec_outs['instances'].pred_masks.cpu().numpy()
    filter_func = lambda x: x.item() in coi
    src_coi, rec_coi = list(map(filter_func, src_outs['instances'].pred_classes)), list(map(filter_func, rec_outs['instances'].pred_classes))
    src_masks, rec_masks = src_masks[src_coi], rec_masks[rec_coi]
    if len(src_masks) != len(rec_masks):
        return np.nan
    values, weights = [], []
    for i in range(len(src_masks)):
        src_mask, rec_mask = crop_image(src_masks[i]), crop_image(rec_masks[i])
        rec_mask = cv2.resize(rec_mask.astype(np.uint8) * 255, src_mask.shape[::-1])
        _, rec_mask = cv2.threshold(rec_mask, 127, 255, cv2.THRESH_BINARY)
        weights.append(np.sum(src_mask))
        values.append(jaccard_score(src_mask, rec_mask, average="micro"))
    return np.average(values, weights)
    