import os
import cv2
from main import init_detectron, create_mask, resize
from tqdm import tqdm
import torch
from outpainter_net.net import OutpainterNet
import sys
import json
import random

sys.path.insert(0, "/mnt/c/Users/proto/repos/image-outpainting")
import outpainting

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_PATH = './data/'

def main():
    with open('data/selected_images.json', 'r') as f:
        paths = json.load(f)
    
    img_names = os.listdir(DATA_PATH)
    predictor, coi = init_detectron()
    img_names.remove('results')

    model1 = OutpainterNet().to(device)
    model1.load_state_dict(torch.load("outpainter_net/state.pt"))
    model1.eval()

    model2 = outpainting.load_model('generator_final.pt')
    model2.eval()

    for i, img_name in tqdm(enumerate(random.sample(paths, 5))):
        if os.path.isfile(img_name):
            image = cv2.imread(img_name)
            target_dims = (256, 320)
            
            mask = create_mask(predictor, coi, image)
            output1 = resize(image, target_dims, None)[0].astype(int)
            output2 = resize(image, target_dims, mask)[0].astype(int)
            output3 = model1.perform_outpaint(image, target_dims, device)
            output4 = (outpainting.perform_outpaint(model2, image)[1] * 255).astype(int)[64:320, 32:352]

            cv2.imwrite(DATA_PATH + 'results/' + str(i) + '_unmasked.png', output1)
            cv2.imwrite(DATA_PATH + 'results/' + str(i) + '_masked.png', output2)
            cv2.imwrite(DATA_PATH + 'results/' + str(i) + '_outpaint.png', output3)
            cv2.imwrite(DATA_PATH + 'results/' + str(i) + '_blended.png', output4)

if __name__ == '__main__':
    main()
