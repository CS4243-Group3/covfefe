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
    print(p1)

    return (features, p1)

def track_features(old_gray, new_gray, features):
    # old_gray = cv2.GaussianBlur(old_gray, (3, 3), 1).T
    # new_gray = cv2.GaussianBlur(new_gray, (3, 3), 1).T

    # Calculative partial derivatives wrt x, y & time
    kernel = np.array([[-1, 0, 1]]) / 2.
    I_x = convolve2d(old_gray, kernel, 'same')
    I_y = convolve2d(old_gray, kernel.T, 'same')
    I_t = old_gray - new_gray

    I_xx = I_x * I_x
    I_xy = I_x * I_y
    I_yy = I_y * I_y

    window_size = 15
    w = window_size // 2
    num_iterations = 1

    ones = np.ones((window_size, window_size))

    weighted_I_xx = convolve2d(I_xx, ones, 'same')
    weighted_I_xy = convolve2d(I_xy, ones, 'same')
    weighted_I_yy = convolve2d(I_yy, ones, 'same')

    good_old = []
    new_features = []

    for feature in features:
        idx = feature[0].astype(np.int)

        left = max(0, idx[0] - w)
        right = min(old_gray.shape[0], idx[0] + w)
        top = max(0, idx[1] - w)
        bottom = min(old_gray.shape[1], idx[1] + w)

        i_x = I_x[top:bottom+1, left:right+1]
        i_y = I_y[top:bottom+1, left:right+1]
        d = np.zeros(2)
        err = False


        for _ in range(num_iterations):

            try:
                xy = (idx + d).astype(np.int)
                Z = np.array([[weighted_I_xx[xy[1], xy[0]], weighted_I_xy[xy[1], xy[0]]],
                              [weighted_I_xy[xy[1], xy[0]], weighted_I_yy[xy[1], xy[0]]]])

                dx = d[0].astype(np.int)
                dy = d[1].astype(np.int)
                i1 = old_gray[top:bottom+1, left:right+1]
                i2 = new_gray[top+dy:bottom+1+dy, left+dx:right+1+dx]

                i_t = i1 - i2
                i_tx = i_t * i_x
                i_ty = i_t * i_y

                b = np.array([np.sum(i_tx.flatten()), np.sum(i_ty.flatten())])
                d += np.linalg.lstsq(Z, b)[0]
                # print(d, idx + d)
            except:
                err = True
                break

        if err:
            continue

        good_old.append(idx)
        new_features.append(feature[0] + d)
    new_features = np.array(new_features).astype(np.float32)
    #print(new_features)
    return (np.array(good_old), new_features)

