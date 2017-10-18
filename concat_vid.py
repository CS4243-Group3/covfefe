import cv2
import numpy as np

before = cv2.VideoCapture('first_before.mp4')
after = cv2.VideoCapture('test.mp4')

framerate = before.get(cv2.CAP_PROP_FPS)
vid_width = int(before.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(before.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_stitch.mp4', fourcc, framerate, (vid_width, vid_height))

while True:
    ret, frame = before.read()
    if not ret:
        break

    cv2.imshow('frame', frame)
    out.write(frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

while True:
    ret, frame = after.read()
    if not ret:
        break

    cv2.imshow('frame', frame)
    out.write(frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
before.release()
after.release()
out.release()