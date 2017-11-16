import numpy as np
import cv2

cap = cv2.VideoCapture('golf.mp4')
_, frame = cap.read()
cv2.imwrite("first.png", frame)