import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.signal import gaussian
from scipy.interpolate import RectBivariateSpline

BIN = True

def track_features(old_gray, new_gray, features):
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_COUNT, 1, 0.03))

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, features, None, **lk_params)

    # Select good points
    # good_new = p1[st==1]
    # good_old = features[st==1]
    #print(p1)

    return (features, p1)

def track_features(old_gray, new_gray, features):
    # old_gray = cv2.GaussianBlur(old_gray, (3, 3), 1).T
    # new_gray = cv2.GaussianBlur(new_gray, (3, 3), 1).T

    num_pyramid = 1

    window_size = 15
    w = window_size // 2
    epsilon = 0.3
    num_iterations = 1

    pyramid = build_pyramid(old_gray, new_gray, num_pyramid, window_size)
    #guassian_weights = np.matmul(gaussian(window_size, 1)[:, None], gaussian(window_size, 1)[None, :])
    guassian_weights = np.ones((window_size, window_size))

    good_old = []
    new_features = []

    for fid in features:
        print('feature', fid)
        feature = fid.flatten()

        g = np.zeros(2)

        err = False

        for L in range(num_pyramid - 1, -1, -1):

            if err:
                break

            old, new, map = pyramid[L]

            f_I_x = map['f_I_x']
            f_I_y = map['f_I_y']
            f_old = map['f_old']
            f_new = map['f_new']
            f_I_xx = map['f_I_xx']
            f_I_xy = map['f_I_xy']
            f_I_yy = map['f_I_yy']

            g = g * 2.
            gx = g[0]
            gy = g[1]

            idx = feature / 2**L

            left = idx[0] - w
            right = idx[0] + w
            top = idx[1] - w
            bottom = idx[1] + w

            y_range = np.arange(top, bottom+1, 1).astype(np.int)
            x_range = np.arange(left, right+1, 1).astype(np.int)

            i_x = f_I_x(y_range, x_range, grid=True)
            i_y = f_I_y(y_range, x_range, grid=True)
            i_xy = f_I_xy(y_range, x_range, grid=True)
            i_xx = f_I_xx(y_range, x_range, grid=True)
            i_yy = f_I_yy(y_range, x_range, grid=True)

            d = np.zeros(2)

            Z = np.array([[np.sum(i_xx * guassian_weights), np.sum(i_xy * guassian_weights)],
                          [np.sum(i_xy * guassian_weights), np.sum(i_yy * guassian_weights)]])
            print('Z', Z)

            eta_norm = 10
            for _ in range(num_iterations):
            #while eta_norm >= epsilon:
                try:
                    if err:
                        break

                    dx = d[0]
                    dy = d[1]

                    new_gray_y_range = (np.arange(top, bottom + 1, 1) + dy + gy).astype(np.int)
                    new_gray_x_range = (np.arange(left, right + 1, 1) + dx + gx).astype(np.int)
                    i2 = f_new(new_gray_y_range, new_gray_x_range, grid=True)
                    i1 = f_old(y_range, x_range, grid=True)

                    i_t = i1 - i2
                    i_tx = i_t * i_x
                    i_ty = i_t * i_y

                    b = np.array([np.sum(i_tx * guassian_weights), np.sum(i_ty * guassian_weights)])
                    print('b', b)
                    eta = np.linalg.lstsq(Z, b)[0]
                    eta_norm = np.linalg.norm(eta)
                    print('eta', eta_norm)
                    d += eta
                except:
                    err = True

            g += d
            print('d', d)

        if err:
            continue
        print('g', g)
        good_old.append(idx)
        new_features.append(fid + g)
    new_features = np.array(new_features).astype(np.float32)
    print(new_features)
    return (np.array(good_old), new_features)

def build_pyramid(old_gray, new_gray, num_pyramid, window_size):
    num_pyramid = max(1, num_pyramid)
    pyramid = [(old_gray, new_gray, build_derivatives(old_gray, new_gray, window_size))]
    for i in range(1, num_pyramid):
        old, new, _ = pyramid[i-1]
        old = cv2.GaussianBlur(old, (3, 3), 1.5)
        old = cv2.resize(old, (old.shape[0] // 2, old.shape[1] // 2))
        new = cv2.GaussianBlur(new, (3, 3), 1.5)
        new = cv2.resize(new, (new.shape[0] // 2, new.shape[1] // 2))
        pyramid.append((old, new, build_derivatives(old, new, window_size)))
    return pyramid

def build_derivatives(old_gray, new_gray, window_size):
    window_size = window_size // 2 + 1

    # Calculative partial derivatives wrt x, y & time
    kernel = np.array([[-1, 0, 1]]) / 2.
    I_x = convolve2d(old_gray, kernel, 'same')
    I_y = convolve2d(old_gray, kernel.T, 'same')

    I_xx = I_x * I_x
    I_xy = I_x * I_y
    I_yy = I_y * I_y

    if BIN:
        f_old = lambda y_range, x_range, grid: old_gray[y_range, x_range]
        f_new = lambda y_range, x_range, grid:  new_gray[y_range, x_range]
        f_I_xx = lambda y_range, x_range, grid:  I_xx[y_range, x_range]
        f_I_xy = lambda y_range, x_range, grid:  I_xy[y_range, x_range]
        f_I_yy = lambda y_range, x_range, grid:  I_yy[y_range, x_range]
        f_I_x = lambda y_range, x_range, grid:  I_x[y_range, x_range]
        f_I_y = lambda y_range, x_range, grid:  I_y[y_range, x_range]

    else:
        y_coords = np.arange(old_gray.shape[0])
        x_coords = np.arange(old_gray.shape[1])
        f_old = RectBivariateSpline(y_coords, x_coords, old_gray)
        f_new = RectBivariateSpline(y_coords, x_coords, new_gray)
        f_I_xx = RectBivariateSpline(y_coords, x_coords, I_xx)
        f_I_xy = RectBivariateSpline(y_coords, x_coords, I_xy)
        f_I_yy = RectBivariateSpline(y_coords, x_coords, I_yy)
        f_I_x = RectBivariateSpline(y_coords, x_coords, I_x)
        f_I_y = RectBivariateSpline(y_coords, x_coords, I_y)

    return {
        'f_old': f_old,
        'f_new': f_new,
        'f_I_x': f_I_x,
        'f_I_y': f_I_y,
        'f_I_xx': f_I_xx,
        'f_I_xy': f_I_xy,
        'f_I_yy': f_I_yy
    }
