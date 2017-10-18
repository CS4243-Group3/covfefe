import cv2
import numpy as np

cup = cv2.VideoCapture('dat.mp4')
bg2 = cv2.VideoCapture('dat_bg2.mp4')

framerate = cup.get(cv2.CAP_PROP_FPS)
vid_width = int(cup.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cup.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dat_output.mp4', fourcc, framerate, (vid_width, vid_height))

bg = cv2.VideoCapture('dat_bg.mp4')
ret_bg, frame_bg = bg.read()

while True:
    old_frame = frame_bg
    ret_bg, frame_bg = bg.read()
    if not ret_bg:
        break
frame_bg = old_frame

ret_bg2, frame_bg2 = bg2.read()

def l2_diff(fc, bg):
    diff = np.linalg.norm(fc.astype(np.float32) - bg.astype(np.float32), axis=2)
    return diff

i = 0
while True:
    i += 1

    ret_cup, frame_cup = cup.read()
    if not ret_cup:
        break

    diff = l2_diff(frame_cup, frame_bg)
    mask = diff > 30
    frame = frame_bg2.copy(); frame[mask != 0] = frame_cup[mask != 0]
    
    cv2.imshow('frame', frame)
    out.write(frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
bg.release()
bg2.release()
cup.release()
out.release()
