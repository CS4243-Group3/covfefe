import cv2
import numpy as np
import math
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from matplotlib.path import Path

cup = cv2.VideoCapture('plus_ultra.mts')
bg2 = cv2.VideoCapture('plus_ultra_bg2.mts')

framerate = cup.get(cv2.CAP_PROP_FPS)
vid_width = int(cup.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cup.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_ultra_output.mp4', fourcc, framerate, (vid_width, vid_height))

"""
bg = cv2.VideoCapture('notebook_bg.mp4')
ret_bg, frame_bg = bg.read()

while True:
    old_frame = frame_bg
    ret_bg, frame_bg = bg.read()
    if not ret_bg:
        break
frame_bg = old_frame

bg.release()
"""

frame_count = cup.get(cv2.CAP_PROP_FRAME_COUNT)
bg_frame = frame_count - 3 * framerate

cup.set(cv2.CAP_PROP_POS_FRAMES, int(bg_frame))

ret_cup, frame_bg = cup.read()
cup.release()

cup = cv2.VideoCapture('plus_ultra.mts')
cup.set(cv2.CAP_PROP_POS_FRAMES, 550)

ret_bg2, frame_bg2 = bg2.read()
cv2.imwrite('plus_ultra_bg2.png', frame_bg2)


def l2_diff(fc, bg):
    diff = np.linalg.norm(fc.astype(np.float32) - bg.astype(np.float32), axis=2)
    return diff


def get_moving_part_in_bounding_box(moving_part_mask, bounding_box_top_left, bounding_box_bottom_right):
    left = bounding_box_top_left[0]
    top = bounding_box_top_left[1]
    right = bounding_box_bottom_right[0]
    bottom = bounding_box_bottom_right[1]

    return moving_part_mask[left:right, top:bottom]


def gen_ellipse_curve(ellipse_x, ellipse_y_radius):
    ellipse_x_radius = int(ellipse_x / 2)
    y = np.zeros(ellipse_x)

    for i, x in enumerate(range(-ellipse_x_radius, ellipse_x_radius)):
        theta = math.acos(x / ellipse_x_radius)
        y[i] = ellipse_y_radius * math.sin(theta)

    return y.astype(int)


def cutoff_contour(moving_part_bounding_box, arm_cutoff_left_bounding_box, arm_cutoff_right_bounding_box, bottom, top, center_y):
    # moving_part_bounding_box[center_y:(bottom - top), :] = False
    ellipse_curve = gen_ellipse_curve(arm_cutoff_right_bounding_box - arm_cutoff_left_bounding_box, 15)
    for x in range(moving_part_bounding_box.shape[1]):
        if x < arm_cutoff_left_bounding_box or x >= arm_cutoff_right_bounding_box:
            moving_part_bounding_box[center_y:(bottom - top), x] = False
        else:
            ellipse_x_displace = x - arm_cutoff_left_bounding_box
            moving_part_bounding_box[center_y + ellipse_curve[ellipse_x_displace]:(bottom - top), x] = False


def cut_moving_part_off_bounding_box(moving_part_mask, bounding_box_top_left, bounding_box_bottom_right):
    left = bounding_box_top_left[0]
    top = bounding_box_top_left[1]
    right = bounding_box_bottom_right[0]
    bottom = bounding_box_bottom_right[1]

    center_y = int((bottom - top) / 2)

    # Clean the parts below
    moving_part_mask[top + center_y + 15:, 0:right] = False

    # Moving part in bounding box?
    moving_part_bounding_box = moving_part_mask[top:bottom, left:right]
    if not np.any(moving_part_bounding_box):
        return None

    # Produce box to blur for
    arm_cutoff_line_indices = np.nonzero(moving_part_mask[top + center_y, left:right])[0]
    if len(arm_cutoff_line_indices) == 0:
        # Entered bounding box, but did not pass center of bounding box
        return None
    arm_cutoff_left = left + arm_cutoff_line_indices[0]
    arm_cutoff_right = left + arm_cutoff_line_indices[-1]
    blur_box = [[arm_cutoff_left, top + center_y - 3], [arm_cutoff_right, top + center_y + 4]]

    # Cutoff
    cutoff_contour(moving_part_bounding_box, arm_cutoff_line_indices[0], arm_cutoff_line_indices[-1], bottom, top,
                   center_y)
    moving_part_mask[top:bottom, left:right] = moving_part_bounding_box

    return blur_box


def gausswin(winsize, sigma):
    filter = np.zeros(winsize)
    filter[int(filter.shape[0]/2), int(filter.shape[1]/2)] = 1.
    return gaussian_filter(filter, sigma=sigma)


def blur_moving_part_bounding_box_interface(blur_box, frame):
    if blur_box is None:
        return

    top_left = blur_box[0]
    bottom_right = blur_box[1]

    left = top_left[0]
    top = top_left[1]
    right = bottom_right[0]
    bottom = bottom_right[1]

    box_to_blur = frame[top-2:bottom+2, left-2:right+2]
    kernel = gausswin([5,5], 0.6)

    for channel in range(3):
        blurred_box = convolve2d(box_to_blur[:, :, channel], kernel, 'same').astype(np.uint8)
        blurred_box = blurred_box[2:-2, 2:-2]
        frame[top:bottom, left:right, channel] = blurred_box

    shadow_line_blur = frame[887-2:929+2, 746-2:825+2]
    kernel = gausswin([5, 5], 1.5)
    for channel in range(3):
        blurred_box = convolve2d(shadow_line_blur[:, :, channel], kernel, 'same').astype(np.uint8)
        blurred_box = blurred_box[2:-2, 2:-2]
        frame[887:929, 746:825, channel] = blurred_box


def create_grace_triangle(triangle_points):
    verts = triangle_points + [(0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)

    x, y = np.meshgrid(np.arange(vid_width), np.arange(vid_height))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    grid = path.contains_points(points)
    grid = grid.reshape((vid_height, vid_width))

    return grid


def create_grace_shape(shape_points):
    verts = shape_points + [(0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    path = Path(verts, codes)

    x, y = np.meshgrid(np.arange(vid_width), np.arange(vid_height))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    grid = path.contains_points(points)
    grid = grid.reshape((vid_height, vid_width))

    return grid


# grace_triangle = create_grace_triangle([(820, 843), (727, 922), (820, 937)])
# grace_triangle = create_grace_triangle([(820, 843), (769, 887), (819, 898)])
grace_triangle = create_grace_shape([(820, 843), (769, 887), (787, 937), (819, 898)])
i = 0
while True:
    i += 1

    ret_cup, frame_cup = cup.read()
    if not ret_cup:
        breakv

    diff = l2_diff(frame_cup, frame_bg)
    # cv2.imshow('frame', diff.astype(np.uint8))
    mask = diff > 20

    curr_grace_triangle = np.logical_and(grace_triangle, mask)

    bounding_box_top_left = (615, 805)
    bounding_box_bottom_right = (820, 873)
    blur_box = cut_moving_part_off_bounding_box(mask, bounding_box_top_left, bounding_box_bottom_right)

    mask = np.logical_or(mask, curr_grace_triangle)

    frame = frame_bg2.copy(); frame[mask != 0] = frame_cup[mask != 0]
    blur_moving_part_bounding_box_interface(blur_box, frame)

    cv2.imshow('frame', frame)
    out.write(frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
bg2.release()
cup.release()
out.release()
