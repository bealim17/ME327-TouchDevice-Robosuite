import robosuite as suite
from robosuite.controllers import load_composite_controller_config

# Check what controllers are available
import robosuite.controllers as ctrl
print(dir(ctrl))

# Try loading OSC_POSE explicitly
try:
    config = load_composite_controller_config(controller="OSC_POSE")
    print("OSC_POSE loaded:", config)
except Exception as e:
    print("OSC_POSE failed:", e)

try:
    config = load_composite_controller_config(controller="BASIC")
    print("BASIC loaded:", config)
except Exception as e:
    print("BASIC failed:", e)