import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.signal import gaussian

def track_features(old_gray, new_gray, features):
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.03))

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, features, None, **lk_params)

    # Select good points
    # good_new = p1[st==1]
    # good_old = features[st==1]
    # print(p1)

    return (features, p1)

def track_features(old_gray, new_gray, features):
    old_gray = cv2.GaussianBlur(old_gray, (3, 3), 1).T
    new_gray = cv2.GaussianBlur(new_gray, (3, 3), 1).T
    # old_gray = old_gray.T
    # new_gray = new_gray.T

    # Calculative partial derivatives wrt x, y & time
    kernel = np.array([[-1, 1]])
    I_x = convolve2d(old_gray, kernel, 'same')
    I_y = convolve2d(old_gray, kernel.T, 'same')
    I_t = old_gray - new_gray

    I_xx = I_x * I_x
    I_xy = I_x * I_y
    I_yy = I_y * I_y
    I_tx = I_t * I_x
    I_ty = I_t * I_y

    window_size = 15
    ones = np.ones((window_size, window_size))

    weighted_I_xx = convolve2d(I_xx, ones, 'same')
    weighted_I_xy = convolve2d(I_xy, ones, 'same')
    weighted_I_yy = convolve2d(I_yy, ones, 'same')

    weighted_I_tx = convolve2d(I_tx, ones, 'same')
    weighted_I_ty = convolve2d(I_ty, ones, 'same')

    good_old = []
    new_features = []
    for feature in features:
        idx = feature[0].astype(np.int)
        if idx[0] > old_gray.shape[0] or idx[1] > old_gray.shape[1]:
            print(idx)
            continue

        Z = np.array([[weighted_I_xx[idx[0], idx[1]], weighted_I_xy[idx[0], idx[1]]],
                      [weighted_I_xy[idx[0], idx[1]], weighted_I_yy[idx[0], idx[1]]]])
        b = np.array([weighted_I_tx[idx[0], idx[1]], weighted_I_ty[idx[0], idx[1]]])
        d = np.linalg.lstsq(Z, b)[0]
        good_old.append(idx)
        new_features.append(feature[0] + d)
    new_features = np.array(new_features).astype(np.float32)
    # print(new_features)
    return (np.array(good_old), new_features)



