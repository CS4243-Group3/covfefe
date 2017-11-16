import cv2
import math
from tqdm import trange

cap = cv2.VideoCapture('plus_ultra_concat.mp4')

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framerate = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

new_width = 1280
new_height = 960

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_shake.mp4', fourcc, framerate, (new_width, new_height))

r_degree = 40
w_degree = 0
h_degree = 110

r_magnitude = 0.2
w_shake_magnitude = 3
h_shake_magnitude = 5

w_padding = int((vid_width - new_width) / 2)
h_padding = int((vid_height - new_height) / 2)

for frame_count in trange(num_frames):

    if frame_count == 1320:
        r_magnitude *= 2
        w_shake_magnitude *= 3
        h_shake_magnitude *= 3
    if frame_count == 1575:
        r_magnitude /= 2
        w_shake_magnitude /= 3
        h_shake_magnitude /= 3

    ret, frame = cap.read()
    if not ret:
        break

    rotate = r_magnitude * (math.sin(math.radians(r_degree)) + math.sin(math.radians(0.5 * r_degree)))
    M = cv2.getRotationMatrix2D((vid_height / 2, vid_width / 2), rotate, 1)
    frame = cv2.warpAffine(frame, M, (vid_width, vid_height))
    r_degree += 2

    w_shake = int(w_shake_magnitude * (math.sin(math.radians(w_degree)) + math.sin(math.radians(0.2 * w_degree))))
    h_shake = int(h_shake_magnitude * math.sin(math.radians(h_degree)))
    w_degree += 5
    h_degree += 3

    new_frame = frame[h_padding + h_shake:\
                  -h_padding + h_shake,\
                  w_padding + w_shake:\
                  -w_padding + w_shake]

    out.write(new_frame)