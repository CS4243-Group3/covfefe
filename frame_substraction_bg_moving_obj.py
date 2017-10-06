import cv2
import numpy as np

bg = cv2.VideoCapture('bg2.mov')
cup = cv2.VideoCapture('hum.mov')

framerate = bg.get(cv2.CAP_PROP_FPS)
vid_width = int(bg.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(bg.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test.mp4', fourcc, framerate, (vid_width, vid_height))

i = 0
while True:
    i += 1

    ret_bg, frame_bg = bg.read()
    ret_cup, frame_cup = cup.read()
    if not ret_bg or not ret_cup:
        break

    diff = frame_cup - frame_bg
    mask = diff > 0

    frame = frame_cup

    if i >= 20:
        frame[mask == 0] = frame_bg[mask == 0]

        values = frame[mask == 1]
        zeros = np.zeros_like(values)
        end = int(1.3**i)
        zeros[:end] = 1
        np.random.shuffle(zeros)
        wbg = frame_bg[mask == 1]
        values[zeros == 1] = wbg[zeros == 1]
        frame[mask == 1] = values

    cv2.imshow('frame', frame)
    out.write(frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
bg.release()
cup.release()
out.release()
