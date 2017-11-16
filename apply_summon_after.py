import cv2
import numpy as np
from constants import SUMMON_SPRITE_COUNT, SUMMON_SPRITE_SCALE, SUMMON_SPRITE_ALPHA_THRESHOLD, \
    SUMMON_SPRITE_BEGIN_FRAME_INDEX, FRAMES_PER_SPRITE, EFFECTS_VIDEO_LENGTH, AFTER_BRIGHTNESS_ADJUSTMENT, \
    SUMMON_SPRITE_DURATION, SUMMON_FADEOUT_DURATION, SUMMON_OFFSET_FROM_CIRCLE_BOX_X, SUMMON_OFFSET_FROM_CIRCLE_BOX_Y, \
    CIRCLE_BOTTOM, CIRCLE_LEFT

# ----------------
# Input & output
# ----------------

after = cv2.VideoCapture('plus_ultra_after.mts')
vid_width = int(after.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(after.get(cv2.CAP_PROP_FRAME_HEIGHT))
framerate = after.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('plus_ultra_after_portal.mp4', fourcc, framerate, (vid_width, vid_height))


# ------------------
# Summoning circle
# ------------------

def load_summon():
    summon_frames = []

    for frame_no in range(SUMMON_SPRITE_COUNT):
        summon = cv2.imread('summon/summon%d.png' % frame_no, cv2.IMREAD_UNCHANGED)
        summon_height = summon.shape[0]
        summon_width = summon.shape[1]
        summon = cv2.resize(summon, (int(summon_width * SUMMON_SPRITE_SCALE), int(summon_height * SUMMON_SPRITE_SCALE)))
        summon_frames.append(summon)

    return summon_frames


summon_bottom = CIRCLE_BOTTOM + SUMMON_OFFSET_FROM_CIRCLE_BOX_Y
summon_left = CIRCLE_LEFT + SUMMON_OFFSET_FROM_CIRCLE_BOX_X


def apply_summon(frame, summon_frames, i):
    i -= summon_after_begin_frame_index + SUMMON_SPRITE_DURATION - SUMMON_FADEOUT_DURATION

    summon_frame = summon_frames[(i % (SUMMON_SPRITE_COUNT * FRAMES_PER_SPRITE)) // FRAMES_PER_SPRITE]
    summon_height = summon_frame.shape[0]
    summon_width = summon_frame.shape[1]

    summon_mask = summon_frame[:, :, 3] > SUMMON_SPRITE_ALPHA_THRESHOLD
    applied_box = frame[summon_bottom-summon_height:summon_bottom, summon_left:summon_left+summon_width]
    if i > 0:
        applied_box[summon_mask] = (
        (i / SUMMON_FADEOUT_DURATION) * applied_box[summon_mask] + ((SUMMON_FADEOUT_DURATION - i) / SUMMON_FADEOUT_DURATION) * summon_frame[:, :, :3][summon_mask]).astype(np.uint8)
    else:
        applied_box[summon_mask] = summon_frame[:, :, :3][summon_mask]

    frame[summon_bottom-summon_height:summon_bottom, summon_left:summon_left+summon_width] = applied_box


# ------------------------------------
# Application of circle to raw video
# ------------------------------------

summon_frames = load_summon()
summon_after_begin_frame_index = ((EFFECTS_VIDEO_LENGTH - SUMMON_SPRITE_BEGIN_FRAME_INDEX) % SUMMON_SPRITE_COUNT) + FRAMES_PER_SPRITE
i = summon_after_begin_frame_index
while True:
    ret_after, frame_after = after.read()
    frame_after = (AFTER_BRIGHTNESS_ADJUSTMENT * frame_after).astype(np.uint8)
    if not ret_after:
        break

    if i < summon_after_begin_frame_index + SUMMON_SPRITE_DURATION:
        apply_summon(frame_after, summon_frames, i)

    cv2.imshow('frame', frame_after)
    out.write(frame_after)

    i += 1

cv2.destroyAllWindows()
after.release()
out.release()
