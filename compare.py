import json
import os
import random
import sys
from tqdm.autonotebook import tqdm

SELECTION_FILENAME = 'data/selected_images.json'
OUTPAINTING_PATH = '/mnt/c/Users/proto/repos/image-outpainting'

sys.path.append(OUTPAINTING_PATH)

import forward2
import recompose

def main(img_num = 1):
    with open(SELECTION_FILENAME, 'r') as f:
        paths = json.load(f)
    paths_sample = random.sample(paths, img_num)
    for i in tqdm(range(len(paths_sample))):
        path = paths_sample[i]
        print(path)
        forward2.main(path, os.path.join('data/results', str(i) + '_extrapolated.jpg'))
        recompose.main(path, os.path.join('data/results', str(i) + '_carved.jpg'), False)
        recompose.main(path, os.path.join('data/results', str(i) + '_recomposed.jpg'), True)

if __name__ == '__main__':
    main(int(sys.argv[1]))