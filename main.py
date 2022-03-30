import cv2
from skimage.segmentation import slic, find_boundaries
import numpy as np
from seam_carving import seam_carve

def segment(image):
    return slic(image, n_segments = 5, sigma = 2, start_label = 0, compactness=5)

def get_neighbour_coords(y, x, height, width):
    # if y > 0:
    #     yield y - 1, x
    # if x > 0:
    #     yield y, x - 1
    if y < height - 1:
        yield y + 1, x
    if x < width - 1:
        yield y, x + 1

def create_graph(segments):
    # boundaries = find_boundaries(segments)
    # height, width = segments.shape
    # edges = []
    # for y, x in np.stack(np.where(boundaries == 1)).T:
    #     seg1 = segments[y, x]
    #     for ny, nx in get_neighbour_coords(y, x, height, width):
    #         seg2 = segments[ny, nx]
    #         if seg1 != seg2:
    #             if seg1 > seg2:
    #                 if (seg2, seg1) not in edges:
    #                     edges.append((seg2, seg1))
    #             elif (seg1, seg2) not in edges:
    #                 edges.append((seg1, seg2))
    
    bounding_boxes = None
    for i in np.unique(segments):
        coords = np.where(segments == i)
        pts = np.array([[[coords[0].min(), coords[1].min()], [coords[0].max() + 1, coords[1].max() + 1]]])
        if bounding_boxes is not None:
            bounding_boxes = np.append(bounding_boxes, pts, axis=0)
        else:
            bounding_boxes = pts
    
    return bounding_boxes

def recompose_graph(segments, bounding_boxes, target_dims):
    source_dims = segments.shape
    y_scale, x_scale = target_dims[0] / source_dims[0], target_dims[1] / source_dims[1]
    new_bounding_boxes = np.zeros_like(bounding_boxes)
    new_bounding_boxes[...,0] = np.rint(bounding_boxes[...,0] * y_scale)
    new_bounding_boxes[...,1] = np.rint(bounding_boxes[...,1] * x_scale)
    return new_bounding_boxes

def carve_seams(image, segments, bounding_boxes, new_bounding_boxes, target_dims):
    new_img = np.zeros((target_dims[0], target_dims[1], 3), dtype=np.uint8)
    for i in np.unique(segments):
        mask = segments == i
        bound_mask = ~mask[bounding_boxes[i][0][0]:bounding_boxes[i][1][0], bounding_boxes[i][0][1]:bounding_boxes[i][1][1]]
        sub_img = image[bounding_boxes[i][0][0]:bounding_boxes[i][1][0], bounding_boxes[i][0][1]:bounding_boxes[i][1][1]]
        d = (new_bounding_boxes[i][1] - new_bounding_boxes[i][0]) - (bounding_boxes[i][1] - bounding_boxes[i][0])
        carved_sub_img, carved_mask = seam_carve(sub_img, d[0], d[1], bound_mask)
        new_mask = np.zeros(target_dims, dtype=bool)
        new_mask[new_bounding_boxes[i][0][0]:new_bounding_boxes[i][1][0], new_bounding_boxes[i][0][1]:new_bounding_boxes[i][1][1]] = ~carved_mask.astype(bool)
        new_img[new_mask,:] = carved_sub_img[~carved_mask.astype(bool)]
    return new_img

def pipeline(image, target_dims):
    segments = segment(image)
    boxes = create_graph(segments)
    target_boxes = recompose_graph(segments, boxes, target_dims)
    output = carve_seams(image, segments, boxes, target_boxes, target_dims)
    return output
