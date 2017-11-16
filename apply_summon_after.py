import cv2
import numpy as np

after = cv2.VideoCapture('plus_ultra_after.mts')

vid_width = int(after.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(after.get(cv2.CAP_PROP_FRAME_HEIGHT))

framerate = after.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_ultra_after_portal.mp4', fourcc, framerate, (vid_width, vid_height))


def load_summon():
    summon_frames = []

    for frame_no in range(45):
        summon = cv2.imread('summon/summon%d.png' % frame_no, cv2.IMREAD_UNCHANGED)
        summon_height = summon.shape[0]
        summon_width = summon.shape[1]
        summon = cv2.resize(summon, (int(summon_width * 1.75), int(summon_height * 1.75)))
        summon_frames.append(summon)

    return summon_frames


def apply_summon(frame, summon_frames, i):
    summon_frame = summon_frames[(i % 90) // 2]
    summon_height = summon_frame.shape[0]
    summon_width = summon_frame.shape[1]

    summon_mask = summon_frame[:, :, 3] > 100

    applied_box = frame[933-summon_height:933, 525:525+summon_width]
    applied_box[summon_mask] = summon_frame[:, :, :3][summon_mask]
    frame[933-summon_height:933, 525:525+summon_width] = applied_box


summon_frames = load_summon()
i = 12  # Previous video ended at 6th frame of animation
while True:
    ret_after, frame_after = after.read()
    frame_after = (0.98 * frame_after).astype(np.uint8)
    if not ret_after:
        break

    if i < 215:
        apply_summon(frame_after, summon_frames, i)

    cv2.imshow('frame', frame_after)
    out.write(frame_after)

    i += 1

cv2.destroyAllWindows()
after.release()
out.release()
