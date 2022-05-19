import os
import cv2
from main import init_detectron, create_mask, resize
from tqdm import tqdm

DATA_PATH = './data/'

def main():
    img_names = os.listdir(DATA_PATH)
    predictor, coi = init_detectron()
    img_names.remove('results')
    for img_name in tqdm(img_names):
        if os.path.isfile(DATA_PATH + img_name):
            image = cv2.imread(DATA_PATH + img_name)
            target_dims = (image.shape[0], int(image.shape[1] * 1.1))
            
            mask = create_mask(predictor, coi, image)
            output1 = resize(image, target_dims, None)
            output2 = resize(image, target_dims, mask)

            cv2.imwrite(DATA_PATH + 'results/' + img_name[:-4] + '_unmasked.png', output1)
            cv2.imwrite(DATA_PATH + 'results/' + img_name[:-4] + '_masked.png', output2)

if __name__ == '__main__':
    main()
