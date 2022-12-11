from skimage.measure import label
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

def unify_sizes(img1, img2):
    ydiff = img2.shape[0] - img1.shape[0]
    xdiff = img2.shape[1] - img1.shape[1]
    padding = np.array([[np.fix(ydiff / 2), ydiff - np.fix(ydiff / 2)],
                        [np.fix(xdiff / 2), xdiff - np.fix(xdiff / 2)]]).astype(int)
    img1 = np.pad(img1, np.maximum(padding, 0))
    img2 = np.pad(img2, np.maximum(-padding, 0))
    return img1, img2

def foreground_score(src_mask, rec_mask):
    src_bin, rec_bin = src_mask >= .5, rec_mask >= .5
    src_img, rec_img = label(src_bin.astype(int), connectivity=2), label(rec_bin.astype(int), connectivity=2)
    if src_img.max() != rec_img.max():
        return np.nan
    scores = []
    weights = []
    for i in range(1, src_img.max() + 1):
        src_object, rec_object = crop_image(src_img == i), crop_image(rec_img == i)
        src_object, rec_object = unify_sizes(src_object, rec_object)
        print(src_object.shape, rec_object.shape)
        scores.append(jaccard_score(src_object, rec_object, average="micro"))
        weights.append(np.sum(src_object))
    return np.average(scores, weights=weights)
