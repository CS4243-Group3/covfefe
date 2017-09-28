import numpy as np
import cv2
import argparse
from lucaskanade import track_features

parser = argparse.ArgumentParser(description='Demo Lucas-Kanade Tracking')
parser.add_argument('video_path', help='path to a video file')
args = vars(parser.parse_args())

cap = cv2.VideoCapture(args['video_path'])

# params for Shi-Tomassi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)


# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
_, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
features = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    good_old, good_new = track_features(old_gray, frame_gray, features)

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    features = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
