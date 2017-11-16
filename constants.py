# -----------------------
# Summoning spritesheet
# -----------------------

SUMMON_SPRITE_COUNT = 45
SUMMON_SPRITE_SCALE = 1.75
SUMMON_SPRITE_ALPHA_THRESHOLD = 100
SUMMON_SPRITE_BEGIN_FRAME_INDEX = 25
FRAMES_PER_SPRITE = 2
SUMMON_SPRITE_DURATION = 213

SUMMON_FADEIN_DURATION = 100
SUMMON_FADEOUT_DURATION = 10

# --------------------
# Stable positioning
# --------------------

CIRCLE_TOP = 805
CIRCLE_BOTTOM = 873
CIRCLE_LEFT = 615
CIRCLE_RIGHT = 820

SUMMONING_CIRCLE_X = 1128
SUMMONING_CIRCLE_Y = 478

SUMMON_OFFSET_FROM_CIRCLE_BOX_X = -90
SUMMON_OFFSET_FROM_CIRCLE_BOX_Y = 60

# Results of tracking while zooming
SUMMONING_CIRCLE_X_FINAL = 646.28063965
SUMMONING_CIRCLE_Y_FINAL = 822.14111328

# ---------
# Zooming
# ---------

BEGIN_ZOOMIN_FRAME = 1250
ZOOMIN_FRAME_COUNT = 100
ZOOMOUT_FRAME_COUNT = 50
ZOOMIN_ZOOMOUT_FRAMEDIFF = 300
MAX_ZOOM = 4
ASPECT_RATIO_HEIGHT = 9
ASPECT_RATIO_WIDTH = 16

# ----------
# Trimming
# ----------

EFFECTS_VIDEO_LENGTH = 3700
EFFECTS_VIDEO_BEGIN_FRAME = 550

# ------------------------------------
# Foreground - background extraction
# ------------------------------------

ELLIPSE_Y = 15
FOREGROUND_THRESHOLD = 20

# --------------------
# Shadow fine-tuning
# --------------------

ARM_SHADOW_AREA_POINTS = [(820, 843), (769, 887), (787, 937), (819, 898)]
SUMMON_OVER_SHADOW_AREA_POINTS = [(925, 800), (849, 819), (769, 887), (849, 944), (925, 955)]

# ------------------------------
# Miscellaneous postprocessing
# ------------------------------

SHADOW_BLUR_TOP = 887
SHADOW_BLUR_BOTTOM = 929
SHADOW_BLUR_LEFT = 746
SHADOW_BLUR_RIGHT = 825

AFTER_BRIGHTNESS_ADJUSTMENT = 0.98
