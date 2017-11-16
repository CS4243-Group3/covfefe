import numpy as np
import cv2
from lucaskanade import track_features
from constants import MAX_ZOOM, ASPECT_RATIO_HEIGHT, ASPECT_RATIO_WIDTH, BEGIN_ZOOMIN_FRAME, \
    SUMMONING_CIRCLE_X, SUMMONING_CIRCLE_Y, ZOOMIN_FRAME_COUNT, ZOOMIN_ZOOMOUT_FRAMEDIFF


# ----------------
# Input & output
# ----------------

cap = cv2.VideoCapture('plus_ultra_before.mts')
vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framerate = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_ultra_before_output.mp4', fourcc, framerate, (vid_width, vid_height))


# -----------------
# Zoom-in helpers
# -----------------

def bb(center, i):
    """
    Generates the bounding box for the zoom.
    """
    w = vid_width - min(i, ZOOMIN_FRAME_COUNT) * (vid_width - vid_width // MAX_ZOOM) / ZOOMIN_FRAME_COUNT
    unit = w // ASPECT_RATIO_WIDTH
    w = unit * ASPECT_RATIO_WIDTH
    h = unit * ASPECT_RATIO_HEIGHT
    x_min = max(0, center[0] - w / 2)
    x_max = center[0] + w - (center[0] - x_min)
    y_min = max(0, center[1] - h / 2)
    y_max = center[1] + h - (center[1] - y_min)

    return list(map(int, [x_min, x_max, y_min, y_max]))


def interpolate_center(feature_coordinates, img_center, i):
    """
    This method introduced to avoid jumping the two frames before
    and after zooming began.
    """
    if i > ZOOMIN_FRAME_COUNT:
        return feature_coordinates

    vector_to_feature = feature_coordinates - img_center
    vector_to_feature_interpolate = vector_to_feature * i / ZOOMIN_FRAME_COUNT
    return img_center + vector_to_feature_interpolate


# ------------------------------------------
# Zoom into each frame of the original video
# ------------------------------------------

# Fast forward
cap.set(cv2.CAP_PROP_POS_FRAMES, BEGIN_ZOOMIN_FRAME)

# First frame to track
_, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
i = 0

# Identify path to tracked feature
features = np.array([[[SUMMONING_CIRCLE_X, SUMMONING_CIRCLE_Y]]]).astype(np.float32)
img_center = [vid_width / 2, vid_height / 2]

while True:
    ret, frame = cap.read()
    if not ret or i > ZOOMIN_ZOOMOUT_FRAMEDIFF:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track towards a point on the summoning circle
    good_old, good_new = track_features(old_gray, frame_gray, features, num_pyramid=5, drop_weak=False)
    zoom_center = interpolate_center(good_new[0], img_center, i)
    minX, maxX, minY, maxY = bb(zoom_center, i)

    # Zoom towards the point
    img = np.zeros((maxY - minY, maxX - minX, 3))
    img = frame[minY:maxY+1, minX:maxX+1]
    img = cv2.resize(img, (vid_width, vid_height))

    cv2.imshow('frame', img)
    out.write(img)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    features = good_new.reshape(-1, 1, 2)
    i += 1

# ---------------------------------------
# Final position of the tracked feature
# ---------------------------------------
print(good_new[0])

cv2.destroyAllWindows()
cap.release()
out.release()
