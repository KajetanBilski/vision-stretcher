from sklearn.metrics import accuracy_score
from main import create_mask
from seam_carving import seam_carve
import numpy as np

def pixel_to_number(pixel):
    return (pixel[0] << 16) + (pixel[1] << 8) + pixel[2]

def foreground_accuracy_score(image, src, mask, src_mask):
    content = np.apply_along_axis(pixel_to_number, 1, image[mask == 255.].astype(int))
    src_content = np.apply_along_axis(pixel_to_number, 1, src[src_mask == 255.].astype(int))
    return accuracy_score(src_content, content)

def test_for_image(img, target_dims, predictor, coi):
    results = {}
    mask = create_mask(predictor, coi, img)
    output, new_mask = seam_carve(img, target_dims[0] - img.shape[0], target_dims[1] - img.shape[1], mask)
    results['augmented'] = foreground_accuracy_score(output, img, new_mask, mask)
    output, _ = seam_carve(img, target_dims[0] - img.shape[0], target_dims[1] - img.shape[1])
    results['default'] = foreground_accuracy_score(output, img, new_mask, mask)
    return results
