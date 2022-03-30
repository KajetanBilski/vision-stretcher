import cv2
from skimage.segmentation import slic

def segment(image):
    return slic(image, n_segments = 5, sigma = 4, start_label = 0, compactness=5)

def create_graph():
    pass

def recompose_graph():
    pass

def warp():
    pass

def pipeline(image, target_dims):
    segments = segment(image)
    create_graph()
    recompose_graph()