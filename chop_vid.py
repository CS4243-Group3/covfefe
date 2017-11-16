import cv2

mid = cv2.VideoCapture('plus_ultra_portal_output_zoomed.mp4')

vid_width = int(mid.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(mid.get(cv2.CAP_PROP_FRAME_HEIGHT))

framerate = mid.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_ultra_portal_output_zoomed_chopped.mp4', fourcc, framerate, (vid_width, vid_height))

i = 0
while True:
    ret, frame = mid.read()
    if not ret or i > 3700:
        break

    out.write(frame)
    i += 1

cv2.destroyAllWindows()
mid.release()
out.release()
