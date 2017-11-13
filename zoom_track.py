import numpy as np
import cv2
import argparse
from lucaskanade import track_features

cap = cv2.VideoCapture('trimmed_before.mp4')

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framerate = cap.get(cv2.CAP_PROP_FPS)

# Take first frame and find corners in it
_, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
features = np.array([[437,302], [421,271]])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, framerate, (vid_width, vid_height))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)


#
# def bb(features):
#     minX = np.floor(min(map(lambda x: x[1], features))).astype(np.int)
#     maxX = np.ceil(max(map(lambda x: x[1], features))).astype(np.int)
#     maxY = np.ceil(max(map(lambda x: x[0], features))).astype(np.int)
#     minY = np.floor(min(map(lambda x: x[0], features))).astype(np.int)
#     w = maxX - minX
#     h = maxY - minY
#     oX = 4*w
#     newH = int((w + 2*oX) / vid_width * vid_height)
#     oY = (newH - h) // 2
#     return minX - oX, maxX + oX, minY - oY, maxY + oY
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # calculate optical flow
#     good_old, good_new = track_features(old_gray, frame_gray, features)
#
#     # draw the tracks
#     # for i, (new, old) in enumerate(zip(good_new, good_old)):
#     #     a, b = new.ravel()
#     #     c, d = old.ravel()
#     #     mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
#     #     frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
#     # img = cv2.add(frame, mask)
#
#     minX, maxX, minY, maxY = bb(good_new)
#
#     img = np.zeros((maxX - minX, maxY - minY, 3))
#     img = frame[minX:maxX+1, minY:maxY+1]
#     img = cv2.resize(img, frame.shape[:-1][::-1])
#
#     cv2.imshow('frame', img)
#     out.write(img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     features = good_new.reshape(-1, 1, 2)
#
# cv2.destroyAllWindows()
# cap.release()
# out.release()
