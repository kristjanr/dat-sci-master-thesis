WANDB_ENABLED = False
TRANSFORMATIONS = ['CROP']
ROI_CROP_TOP = 60
ROI_CROP_KEEP_SIZE = False
SEQUENCE_LENGTH = 3

JOYSTICK_STEERING_SCALE = 0.8


TURN_BOOST_TOGGLE_BUTTON = 'A'

# Keys are turning angles which equal or bigger, when positive or equal or smaller, when negative need a boost.
# Values are the throttle increase for during that angle turn.
TURN_BOOST_CONFIG = {0.8: 0.03}   # - when angle is >= 0.8, then add 0.03 to the throttle