import sys
from main import *
import numpy as np
import cv2

TARGET_MULTIPLIER = np.array((1, 1.25))
predictor, coi = init_detectron()

def main(input_path, output_path, use_mask = True):
    image = cv2.imread(input_path)
    target_dims = (np.array(image.shape[:-1]) * TARGET_MULTIPLIER).astype(int)
    if use_mask:
        mask = create_mask(predictor, coi, image)
        cv2.imwrite('mask.jpg', mask)
        output = resize(image, target_dims, mask)
    else:
        output = resize(image, target_dims, None)
    cv2.imwrite(output_path, output)

if __name__ == '__main__':
    main(*sys.argv[1:3])
