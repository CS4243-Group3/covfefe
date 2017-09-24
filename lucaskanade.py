import numpy as np
import cv2

def track_features(old_gray, new_gray, features):
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, features, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = features[st==1]

    return (good_old, good_new)
