import cv2
import numpy as np
import math
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from matplotlib.path import Path
from random import randint

from constants import ELLIPSE_Y, EFFECTS_VIDEO_LENGTH, EFFECTS_VIDEO_BEGIN_FRAME, FOREGROUND_THRESHOLD, \
    SUMMON_SPRITE_COUNT, SUMMON_SPRITE_SCALE, SUMMON_SPRITE_ALPHA_THRESHOLD, SUMMON_SPRITE_BEGIN_FRAME_INDEX, \
    SUMMON_FADEIN_DURATION, FRAMES_PER_SPRITE, SUMMON_OFFSET_FROM_CIRCLE_BOX_X, SUMMON_OFFSET_FROM_CIRCLE_BOX_Y, \
    CIRCLE_TOP, CIRCLE_BOTTOM, CIRCLE_LEFT, CIRCLE_RIGHT, \
    SHADOW_BLUR_TOP, SHADOW_BLUR_BOTTOM, SHADOW_BLUR_LEFT, SHADOW_BLUR_RIGHT, \
    ARM_SHADOW_AREA_POINTS, SUMMON_OVER_SHADOW_AREA_POINTS

# ----------------
# Input & output
# ----------------

cap = cv2.VideoCapture('plus_ultra.mts')
framerate = cap.get(cv2.CAP_PROP_FPS)
vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_ultra_portal_output.mp4', fourcc, framerate, (vid_width, vid_height))

# Grab a frame near the end of the video as the background to extract movement from
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
bg_frame = frame_count - 3 * framerate
cap.set(cv2.CAP_PROP_POS_FRAMES, int(bg_frame))
ret_cap, frame_bg = cap.read()
cap.release()

# Reload the video to start from the beginning
cap = cv2.VideoCapture('plus_ultra.mts')
cap.set(cv2.CAP_PROP_POS_FRAMES, EFFECTS_VIDEO_BEGIN_FRAME)

# Grab a frame from the background video
bg2 = cv2.VideoCapture('plus_ultra_bg2.mts')
ret_bg2, frame_bg2 = bg2.read()

# ------------------------
# Foreground masking-off
# ------------------------

def l2_diff(fc, bg):
    diff = np.linalg.norm(fc.astype(np.float32) - bg.astype(np.float32), axis=2)
    return diff


# ---------------
# Cutoff of arm
# ---------------

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
    ellipse_curve = gen_ellipse_curve(arm_cutoff_right_bounding_box - arm_cutoff_left_bounding_box, ELLIPSE_Y)
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
    moving_part_mask[top + center_y + ELLIPSE_Y:, 0:right] = False

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


# --------------------------
# Blurring of cutoff parts
# --------------------------

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

    shadow_line_blur = frame[SHADOW_BLUR_TOP-2:SHADOW_BLUR_BOTTOM+2, SHADOW_BLUR_LEFT-2:SHADOW_BLUR_RIGHT+2]
    kernel = gausswin([5, 5], 1.5)
    for channel in range(3):
        blurred_box = convolve2d(shadow_line_blur[:, :, channel], kernel, 'same').astype(np.uint8)
        blurred_box = blurred_box[2:-2, 2:-2]
        frame[SHADOW_BLUR_TOP:SHADOW_BLUR_BOTTOM, SHADOW_BLUR_LEFT:SHADOW_BLUR_RIGHT, channel] = blurred_box


# ------------------
# Particle Effects
# ------------------

class Particle:

    def __init__(self, position, countdown=60, distance=50, alpha=1.0):
        self.position = position # position is y, x
        self.countdown = countdown # in frames
        self.alpha = alpha
        self.travel_distance = distance
        self.speed = distance / countdown
        self.decay = alpha / countdown

    def next_frame(self):
        self.position[0] -= self.speed
        self.countdown -= 1
        self.decrease_alpha()

    def decrease_alpha(self):
        self.alpha -= self.decay
        if(self.alpha < 0):
            self.alpha = 0

    def renew(self, w, h, ref_coord):
        rand_x = ref_coord[1] + randint(0, w)
        rand_y = ref_coord[0] - randint(0, h)
        self.position = [rand_y, rand_x]
        self.countdown = randint(60, 180)
        self.alpha = 1.0
        self.travel_distance = randint(40, 100)
        self.speed = self.travel_distance / self.countdown
        self.decay = self.alpha / self.countdown

def init_particle_system(pool_size):

    pool = []
    w = 330
    h = 110
    ref_coord = [903, 550]

    for i in range(pool_size):
        rand_x = ref_coord[1] + randint(0, w)
        rand_y = ref_coord[0] - randint(0, h)
        spawn_point = [rand_y, rand_x]
        pool.append(Particle(spawn_point))

    return pool

def animate_particles(frame, particle_pool, img):

    w = 330
    h = 210
    ref_coord = [903, 550]

    for particle in particle_pool:
        draw_particle_in_location(frame, particle.position, particle.alpha, img)
        particle.next_frame()
        if(particle.countdown < 0):
            particle.renew(w, h, ref_coord)

def draw_particle_in_location(frame, location, alpha, particle):

    location[0] = (int)(location[0])
    location[1] = (int)(location[1])
    affected_area = frame[location[0]-16:location[0]+16, location[1]-16:location[1]+16]
    white = np.array([255, 255, 255], dtype='uint8')
    '''temp = np.zeros([8,8,3], dtype='uint8')
    for x in range(8):
        for y in range(8):
            temp[y, x] = white'''

    temp = particle;

    temp[0, 0] = affected_area[11, 11]
    temp[0, 7] = affected_area[11, 19]
    temp[7, 0] = affected_area[19, 11]
    temp[7, 7] = affected_area[19, 19]

    affected_area[11:19,11:19] = affected_area[11:19,11:19] * (1.0 - alpha) + temp * alpha

    # Gaussian blur the affected area
    affected_area = cv2.GaussianBlur(affected_area, (5,5), 1.0)

    frame[location[0]-16:location[0]+16, location[1]-16:location[1]+16] = affected_area

def animate_small_particles(frame, particle_pool):

    w = 330
    h = 210
    ref_coord = [903, 550]

    for particle in particle_pool:
        draw_small_particle_in_location(frame, particle.position, particle.alpha)
        particle.next_frame()
        if(particle.countdown < 0):
            particle.renew(w, h, ref_coord)

def draw_small_particle_in_location(frame, location, alpha):

    location[0] = (int)(location[0])
    location[1] = (int)(location[1])
    affected_area = frame[location[0]-16:location[0]+16, location[1]-16:location[1]+16]
    white = np.array([255, 255, 255], dtype='uint8')
    temp = np.zeros([4,4,3], dtype='uint8')
    for x in range(4):
        for y in range(4):
            temp[y, x] = white

    affected_area[13:17,13:17] = affected_area[13:17,13:17] * (1.0 - alpha) + temp * alpha

    # Gaussian blur the affected area
    affected_area = cv2.GaussianBlur(affected_area, (5,5), 0.5)

    frame[location[0]-16:location[0]+16, location[1]-16:location[1]+16] = affected_area

# ---------------
# Summon circle
# ---------------

def load_summon():
    summon_frames = []

    for frame_no in range(SUMMON_SPRITE_COUNT):
        summon = cv2.imread('summon/summon%d.png' % frame_no, cv2.IMREAD_UNCHANGED)
        summon_height = summon.shape[0]
        summon_width = summon.shape[1]
        summon = cv2.resize(summon, (int(summon_width * SUMMON_SPRITE_SCALE), int(summon_height * SUMMON_SPRITE_SCALE)))
        summon_frames.append(summon)

    return summon_frames


summon_bottom = CIRCLE_BOTTOM + SUMMON_OFFSET_FROM_CIRCLE_BOX_Y
summon_left = CIRCLE_LEFT + SUMMON_OFFSET_FROM_CIRCLE_BOX_X

def apply_summon(special_area_mask, apply_special_area, frame, summon_frames, i):
    i -= SUMMON_SPRITE_BEGIN_FRAME_INDEX
    summon_frame = summon_frames[(i % (SUMMON_SPRITE_COUNT * FRAMES_PER_SPRITE)) // FRAMES_PER_SPRITE]
    summon_height = summon_frame.shape[0]
    summon_width = summon_frame.shape[1]

    summon_mask = summon_frame[:, :, 3] > SUMMON_SPRITE_ALPHA_THRESHOLD

    applied_box = frame[summon_bottom-summon_height:summon_bottom, summon_left:summon_left+summon_width]
    grace_box = special_area_mask[summon_bottom-summon_height:summon_bottom, summon_left:summon_left+summon_width]

    if not apply_special_area:
        grace_box = np.invert(grace_box)

    summon_mask = np.logical_and(summon_mask, grace_box)

    if i < SUMMON_FADEIN_DURATION:
        background_alpha_blend = ((SUMMON_FADEIN_DURATION - i) / SUMMON_FADEIN_DURATION) * applied_box[summon_mask]
        summon_alpha_blend = (i / SUMMON_FADEIN_DURATION) * summon_frame[:, :, :3][summon_mask]
        applied_box[summon_mask] = (background_alpha_blend + summon_alpha_blend).astype(np.uint8)
    else:
        applied_box[summon_mask] = summon_frame[:, :, :3][summon_mask]

    frame[summon_bottom-summon_height:summon_bottom, summon_left:summon_left+summon_width] = applied_box


# -------------------------
# Special treatment areas
# -------------------------

def create_special_area_mask(shape_points, end_with_curve):
    verts = shape_points + [(0, 0)]
    codes = [Path.MOVETO] + (len(shape_points) - 1) * [Path.LINETO] + [Path.CLOSEPOLY]
    if end_with_curve:
        codes[-3] = Path.CURVE3
        codes[-2] = Path.CURVE3
    path = Path(verts, codes)

    x, y = np.meshgrid(np.arange(vid_width), np.arange(vid_height))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    grid = path.contains_points(points)
    grid = grid.reshape((vid_height, vid_width))

    return grid


# ------------------------------------------
# Process each frame of original video
# ------------------------------------------

summon_frames = load_summon()
particle_pool_back = init_particle_system(7)
particle_pool_front = init_particle_system(7)
small_particle_pool_back = init_particle_system(7)
small_particle_pool_front = init_particle_system(7)
particle_img = cv2.imread('particle.png', cv2.IMREAD_COLOR)
arm_shadow_area = create_special_area_mask(ARM_SHADOW_AREA_POINTS, end_with_curve=True)
summon_over_shadow_area = create_special_area_mask(SUMMON_OVER_SHADOW_AREA_POINTS, end_with_curve=False)
i = 0
while True:
    i += 1

    ret_cap, frame_cap = cap.read()
    if not ret_cap or i > EFFECTS_VIDEO_LENGTH:
        break

    # Extract moving foreground
    diff = l2_diff(frame_cap, frame_bg)
    mask = diff > FOREGROUND_THRESHOLD

    # Cut arm portions off moving foreground
    bounding_box_top_left = (CIRCLE_LEFT, CIRCLE_TOP)
    bounding_box_bottom_right = (CIRCLE_RIGHT, CIRCLE_BOTTOM)
    blur_box = cut_moving_part_off_bounding_box(mask, bounding_box_top_left, bounding_box_bottom_right)

    # Allow shadow from moving foreground to appear
    curr_arm_shadow_area = np.logical_and(arm_shadow_area, mask)
    mask = np.logical_or(mask, curr_arm_shadow_area)

    frame = frame_bg2.copy()

    # Apply summoning circle below foreground
    if i > SUMMON_SPRITE_BEGIN_FRAME_INDEX:
        apply_summon(summon_over_shadow_area, False, frame, summon_frames, i)

    # Particle system in background
    animate_particles(frame, particle_pool_back, particle_img)
    animate_small_particles(frame, small_particle_pool_back)

    # Apply foreground
    frame[mask != 0] = frame_cap[mask != 0]
    blur_moving_part_bounding_box_interface(blur_box, frame)

    # Apply summoning circle above arm shadow in foreground
    if i > SUMMON_SPRITE_BEGIN_FRAME_INDEX:
        apply_summon(summon_over_shadow_area, True, frame, summon_frames, i)

    # Particle system in foreground
    animate_particles(frame, particle_pool_front, particle_img)
    animate_small_particles(frame, small_particle_pool_front)

    cv2.imshow('frame', frame)
    out.write(frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
bg2.release()
cap.release()
out.release()
