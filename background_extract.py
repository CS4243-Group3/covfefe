import numpy as np
import cv2

cap = cv2.VideoCapture('hakase_lv2.mov')

num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
averages = np.zeros((vid_height, vid_width, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    averages += frame / num_frames

cv2.imwrite('background.png', averages)

cap.release()

# Reopen
cap = cv2.VideoCapture('hakase_lv2.mov')
framerate = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('hakase_foreground.mp4', fourcc, framerate, (vid_width, vid_height))

def compute_pixel_luma_mask(frame):
    frame_expandable = frame.astype(np.float32)
    luma = np.sqrt(frame_expandable[:,:,0] * frame_expandable[:,:,0] + frame_expandable[:,:,1] * frame_expandable[:,:,1] + frame_expandable[:,:,2] * frame_expandable[:,:,2])
    return luma < 100

def extract_moving_guy(frame, bg):
    moving_shape = np.abs(frame.astype(np.float32) - averages)
    luma_mask = compute_pixel_luma_mask(moving_shape)
    frame[luma_mask] = 0
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(extract_moving_guy(frame, averages))

out.release()
cap.release()