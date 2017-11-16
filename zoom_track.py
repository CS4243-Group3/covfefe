import numpy as np
import cv2
from lucaskanade import track_features

cap = cv2.VideoCapture('plus_ultra_before.mts')

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(vid_width, vid_height)
framerate = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_ultra_before_output.mp4', fourcc, framerate, (vid_width, vid_height))

# Fast forward
cap.set(cv2.CAP_PROP_POS_FRAMES, 1250)


# Take first frame and find corners in it
_, old_frame = cap.read()
print(old_frame.shape)
cv2.imwrite('first.png', old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
features = np.array([[[1128, 478]]]).astype(np.float32)


def bb(center, i):
    w = vid_width - min(i, 100) * (vid_width - vid_width // 4) / 100
    unit = w // 16
    w = unit * 16
    h = unit * 9
    x_min = max(0, center[0] - w / 2)
    x_max = center[0] + w - (center[0] - x_min)
    y_min = max(0, center[1] - h / 2)
    y_max = center[1] + h - (center[1] - y_min)

    return list(map(int, [x_min, x_max, y_min, y_max]))


img_center = [vid_width / 2, vid_height / 2]


def interpolate_center(feature_coordinates, img_center, i):
    if i > 100:
        return feature_coordinates

    vector_to_feature = feature_coordinates - img_center
    vector_to_feature_interpolate = vector_to_feature * i / 100
    return img_center + vector_to_feature_interpolate


i = 0
while True:
    ret, frame = cap.read()
    if not ret or i > 300:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    good_old, good_new = track_features(old_gray, frame_gray, features, num_pyramid=5, drop_weak=False)
    zoom_center = interpolate_center(good_new[0], img_center, i)
    minX, maxX, minY, maxY = bb(zoom_center, i)

    img = np.zeros((maxY - minY, maxX - minX, 3))
    img = frame[minY:maxY+1, minX:maxX+1]
    img = cv2.resize(img, (vid_width, vid_height))
    """
    a, b = good_new[0].ravel()
    img = cv2.circle(frame, (a, b), 5, (255, 0, 0), -1)
    """

    cv2.imshow('frame', img)
    out.write(img)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    features = good_new.reshape(-1, 1, 2)
    i += 1

print(good_new[0])

cv2.destroyAllWindows()
cap.release()
out.release()
