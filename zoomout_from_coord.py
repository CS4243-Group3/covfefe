import numpy as np
import cv2

cap = cv2.VideoCapture('plus_ultra_portal_output.mp4')

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(vid_width, vid_height)
framerate = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_ultra_portal_output_zoomed.mp4', fourcc, framerate, (vid_width, vid_height))


def bb(center, i):
    w = vid_width - (50 - i) * (vid_width - vid_width // 4) / 50
    unit = w // 16
    w = unit * 16
    h = unit * 9
    x_min = max(0, center[0] - w / 2)
    x_max = center[0] + w - (center[0] - x_min)
    y_max = min(vid_height, center[1] + h / 2)
    y_min = y_max - h

    return list(map(int, [x_min, x_max, y_min, y_max]))


def interpolate_center(feature_coordinates, img_center, i):
    if i > 50:
        return img_center

    vector_to_feature = feature_coordinates - img_center
    vector_to_feature_interpolate = vector_to_feature * (50 - i) / 50
    return img_center + vector_to_feature_interpolate


feature_coordinates = [646.28063965, 822.14111328]
img_center = np.array([vid_width / 2, vid_height / 2])
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if i < 50:
        zoom_center = interpolate_center(feature_coordinates, img_center, i)
        minX, maxX, minY, maxY = bb(zoom_center, i)

        img = np.zeros((maxY - minY, maxX - minX, 3))
        img = frame[minY:maxY+1, minX:maxX+1]
        img = cv2.resize(img, (vid_width, vid_height))
    else:
        img = frame

    cv2.imshow('frame', img)
    out.write(img)

    i += 1

cv2.destroyAllWindows()
cap.release()
out.release()
