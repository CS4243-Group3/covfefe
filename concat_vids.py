import cv2
import numpy as np

before = cv2.VideoCapture('plus_ultra_before.mts')

before_zoom = cv2.VideoCapture('plus_ultra_before_output.mp4')
mid = cv2.VideoCapture('plus_ultra_portal_output_zoomed_chopped.mp4')
after = cv2.VideoCapture('plus_ultra_after_portal.mp4')

vid_width = int(before.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(before.get(cv2.CAP_PROP_FRAME_HEIGHT))

framerate = before.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_ultra_concat.mp4', fourcc, framerate, (vid_width, vid_height))

i = 0
while True:
    ret, frame = before.read()
    if not ret or i > 1250:
        break

    out.write(frame)

    i += 1

while True:
    ret, frame = before_zoom.read()
    if not ret:
        break

    out.write(frame)

while True:
    ret, frame = mid.read()
    if not ret:
        break

    out.write(frame)

i = 0
while True:
    ret, frame = after.read()
    if not ret:
        break

    out.write(frame.astype(np.uint8))
    i += 1


cv2.destroyAllWindows()
before.release()
before_zoom.release()
mid.release()
after.release()
out.release()
