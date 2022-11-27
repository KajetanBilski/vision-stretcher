

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
    if len(src_masks) != len(rec_masks):
        return 0.
    