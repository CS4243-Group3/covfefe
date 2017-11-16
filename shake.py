import cv2
import math

cap = cv2.VideoCapture('plus_ultra_concat.mp4')

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framerate = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Largest 4:3 resolution that can be cropped from video source
new_width = 1280
new_height = 960

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_shake.mp4', fourcc, framerate, (new_width, new_height))

# Magnitudes of rotation, shaking along width, shaking along height
r_magnitude = 0.2
w_shake_magnitude = 3
h_shake_magnitude = 5

# Degrees for sinusoidal oscillations
# Initialized to different degrees to not have a uniform start state
r_degree = 40
w_degree = 0
h_degree = 110

# Padding between new resolution and old resolution
w_padding = int(math.floor((vid_width - new_width) / 2))
h_padding = int(math.floor((vid_height - new_height) / 2))

# Estimated start and end of zoomed segment
zoomed_start_time = 26.4
zoomed_end_time = 31.5
zoomed_start_frame_count = framerate * zoomed_start_time
zoomed_end_frame_count = framerate * zoomed_end_time

# Modifiers of magnitude values during zoomed segment
r_magnitude_zoom_factor = 2
w_magnitude_zoom_factor = 3
h_magnitude_zoom_factor = 3

for frame_count in range(num_frames):

    # Simulate increased shaking magnitude during zoomed segment
    if frame_count == zoomed_start_frame_count:
        r_magnitude *= r_magnitude_zoom_factor
        w_shake_magnitude *= w_magnitude_zoom_factor
        h_shake_magnitude *= h_magnitude_zoom_factor
    if frame_count == zoomed_end_frame_count:
        r_magnitude /= r_magnitude_zoom_factor
        w_shake_magnitude /= w_magnitude_zoom_factor
        h_shake_magnitude /= h_magnitude_zoom_factor

    ret, frame = cap.read()
    if not ret:
        break

    # Simulate frame rotation about center of frame
    rotate = r_magnitude * (math.sin(math.radians(r_degree)) + math.sin(math.radians(0.5 * r_degree)))
    M = cv2.getRotationMatrix2D((vid_height / 2, vid_width / 2), rotate, 1)
    frame = cv2.warpAffine(frame, M, (vid_width, vid_height))

    # Simulate frame shaking using different sine waves to avoid uniform shake patterns
    w_shake = int(w_shake_magnitude * (math.sin(math.radians(w_degree)) + math.sin(math.radians(0.2 * w_degree))))
    h_shake = int(h_shake_magnitude * math.sin(math.radians(h_degree)))

    # Increment degrees at different steps to avoid uniform shake pattern
    r_degree += 2
    w_degree += 5
    h_degree += 3

    new_frame = frame[h_padding + h_shake:\
                  -h_padding + h_shake,\
                  w_padding + w_shake:\
                  -w_padding + w_shake]

    out.write(new_frame)
    