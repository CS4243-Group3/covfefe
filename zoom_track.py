import numpy as np
import cv2
import argparse
from lucaskanade import track_features
from subprocess import call

call(['rm', 'output.mp4'])

cap = cv2.VideoCapture('fma.mp4')

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(vid_width, vid_height)
framerate = cap.get(cv2.CAP_PROP_FPS)

# Take first frame and find corners in it
_, old_frame = cap.read()
print(old_frame.shape)
cv2.imwrite('first.png', old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
features = np.array([[[751, 486]]]).astype(np.float32)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, framerate, (vid_width, vid_height))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)


#
def bb(center, i):
    w = 1440 - min(i, 40) * (1440 - 480) / 40
    unit = w // 4
    w = unit * 4
    h = unit * 3
    x_min = max(0, center[0] - w / 2)
    x_max = center[0] + w - (center[0] - x_min)
    y_min = max(0, center[1] - h / 2)
    y_max = center[1] + h - (center[1] - y_min)

    return list(map(int, [x_min, x_max, y_min, y_max]))

i = 0
while True:
    ret, frame = cap.read()
    # print(frame.shape)
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    good_old, good_new = track_features(old_gray, frame_gray, features, num_pyramid=5)

    # draw the tracks
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    #     frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    # img = cv2.add(frame, mask)
    # print(good_new)

    minX, maxX, minY, maxY = bb(good_new[0], i)

    img = np.zeros((maxY - minY, maxX - minX, 3))
    img = frame[minY:maxY+1, minX:maxX+1]
    img = cv2.resize(img, (vid_width, vid_height))

    cv2.imshow('frame', img)
    out.write(img)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    features = good_new.reshape(-1, 1, 2)
    i += 1

cv2.destroyAllWindows()
cap.release()
out.release()
