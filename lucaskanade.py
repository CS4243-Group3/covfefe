import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.interpolate import RectBivariateSpline


def track_features(old_gray, new_gray, features, num_pyramid=4, num_iterations=1, window_size=13, drop_weak=True):
    num_pyramid = max(num_pyramid, 1)
    window_size = window_size // 2 * 2 + 1
    num_iterations = max(num_iterations, 1)

    pyramid = build_pyramid(old_gray, new_gray, num_pyramid)

    old_features = []
    new_features = []

    for unflattened_feature in features:
        feature = unflattened_feature.flatten()

        try:
            displacement = get_pyramidal_lk(feature, num_iterations, pyramid, window_size)
            new_feature = (feature + displacement).astype(np.float32)

            # Drop point if not a corner
            if drop_weak and not is_corner(new_feature, pyramid[0][2], window_size // 2):
                continue

            old_features.append(feature)
            new_features.append(new_feature)
        except Exception as e:
            print(str(e))
            continue

    return np.array(old_features), np.array(new_features)


def get_pyramidal_lk(coordinates, num_iterations, pyramid, window_size):
    num_levels = len(pyramid)

    pyramidal_displacement = np.zeros(2)

    for pyramid_level in range(num_levels - 1, -1, -1):
        _, _, f_map = pyramid[pyramid_level]
        level_coordinates = coordinates / 2 ** pyramid_level
        pyramidal_displacement *= 2.
        level_displacement = get_iterative_lk(level_coordinates, num_iterations, f_map, pyramidal_displacement, window_size)
        pyramidal_displacement += level_displacement

    return pyramidal_displacement


def get_iterative_lk(coordinates, num_iterations, f_map, initial_estimate, window_size):
    f_I_x = f_map['f_I_x']
    f_I_y = f_map['f_I_y']
    f_old = f_map['f_old']
    f_new = f_map['f_new']
    f_I_xx = f_map['f_I_xx']
    f_I_xy = f_map['f_I_xy']
    f_I_yy = f_map['f_I_yy']

    radius = window_size // 2

    horizontal_displacement_estimate = initial_estimate[0]
    vertical_displacement_estimate = initial_estimate[1]

    left = coordinates[0] - radius
    right = coordinates[0] + radius
    top = coordinates[1] - radius
    bottom = coordinates[1] + radius
    y_range = np.arange(top, bottom + 1, 1)
    x_range = np.arange(left, right + 1, 1)

    window_I_x = f_I_x(y_range, x_range, grid=True)
    window_I_y = f_I_y(y_range, x_range, grid=True)
    window_I_xy = f_I_xy(y_range, x_range, grid=True)
    window_I_xx = f_I_xx(y_range, x_range, grid=True)
    window_I_yy = f_I_yy(y_range, x_range, grid=True)

    displacement = np.zeros(2)

    Z = np.array([[np.sum(window_I_xx), np.sum(window_I_xy)],
                  [np.sum(window_I_xy), np.sum(window_I_yy)]])

    for _ in range(num_iterations):
        dx = displacement[0]
        dy = displacement[1]

        new_gray_y_range = np.arange(top, bottom + 1, 1) + dy + vertical_displacement_estimate
        new_gray_x_range = np.arange(left, right + 1, 1) + dx + horizontal_displacement_estimate

        window_new = f_new(new_gray_y_range, new_gray_x_range, grid=True)
        window_old = f_old(y_range, x_range, grid=True)

        window_I_t = window_old - window_new
        window_I_tx = window_I_t * window_I_x
        window_I_ty = window_I_t * window_I_y

        b = np.array([np.sum(window_I_tx), np.sum(window_I_ty)])
        eta = np.linalg.lstsq(Z, b)[0]
        displacement += eta

    return displacement


def build_pyramid(old_gray, new_gray, num_pyramid):
    pyramid = [(old_gray, new_gray, build_derivatives(old_gray, new_gray))]
    for i in range(1, num_pyramid):
        old, new, _ = pyramid[i-1]
        old = cv2.GaussianBlur(old, (3, 3), 1.5)
        old = cv2.resize(old, (old.shape[1] // 2, old.shape[0] // 2))
        new = cv2.GaussianBlur(new, (3, 3), 1.5)
        new = cv2.resize(new, (new.shape[1] // 2, new.shape[0] // 2))
        pyramid.append((old, new, build_derivatives(old, new)))
    return pyramid


def build_derivatives(old_gray, new_gray):
    # Calculate partial derivatives wrt x, y & time
    kernel = np.array([[1, 0, -1]]) / 2.
    I_x = convolve2d(old_gray, kernel, 'same')
    I_y = convolve2d(old_gray, kernel.T, 'same')

    I_xx = I_x * I_x
    I_xy = I_x * I_y
    I_yy = I_y * I_y

    height, width = old_gray.shape

    y_coords = np.arange(height)
    x_coords = np.arange(width)
    f_old = get_range_values(y_coords, x_coords, old_gray)
    f_new = get_range_values(y_coords, x_coords, new_gray)
    f_I_xx = get_range_values(y_coords, x_coords, I_xx)
    f_I_xy = get_range_values(y_coords, x_coords, I_xy)
    f_I_yy = get_range_values(y_coords, x_coords, I_yy)
    f_I_x = get_range_values(y_coords, x_coords, I_x)
    f_I_y = get_range_values(y_coords, x_coords, I_y)

    return {
        'f_old': f_old,
        'f_new': f_new,
        'f_I_x': f_I_x,
        'f_I_y': f_I_y,
        'f_I_xx': f_I_xx,
        'f_I_xy': f_I_xy,
        'f_I_yy': f_I_yy
    }


def get_range_values(y_coords, x_coords, values):
    return RectBivariateSpline(y_coords, x_coords, values, kx=2, ky=2)

# References
# https://docs.opencv.org/trunk/d4/d7d/tutorial_harris_detector.html
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
# https://github.com/codeplaysoftware/visioncpp/wiki/Example:-Harris-Corner-Detection
# Why not Tomasi? for performance
def is_corner(coordinates, f_map, radius):
    threshold = 4.
    k = 0.04

    f_I_xx = f_map['f_I_xx']
    f_I_xy = f_map['f_I_xy']
    f_I_yy = f_map['f_I_yy']

    left = coordinates[0] - radius
    right = coordinates[0] + radius
    top = coordinates[1] - radius
    bottom = coordinates[1] + radius
    y_range = np.arange(top, bottom + 1, 1)
    x_range = np.arange(left, right + 1, 1)

    window_I_xy = f_I_xy(y_range, x_range, grid=True)
    window_I_xx = f_I_xx(y_range, x_range, grid=True)
    window_I_yy = f_I_yy(y_range, x_range, grid=True)

    M = np.array([[np.sum(window_I_xx), np.sum(window_I_xy)],
                  [np.sum(window_I_xy), np.sum(window_I_yy)]])

    R = np.linalg.det(M) - k*np.square(np.trace(M))
    if R <= threshold:
        print(R)
    return R > threshold
