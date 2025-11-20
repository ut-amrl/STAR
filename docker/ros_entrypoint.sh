#!/usr/bin/env bash
set -e

# Source ROS
if [ -f "/opt/ros/noetic/setup.bash" ]; then
  source /opt/ros/noetic/setup.bash
fi

# If you later add a workspace at /ws, auto-source it when present
if [ -f "/ws/devel/setup.bash" ]; then
  source /ws/devel/setup.bash
fi

if [ -n "$MOMA_WORKDIR" ] && [ -d "$MOMA_WORKDIR" ]; then
  cd "$MOMA_WORKDIR"
fi
export ROS_PACKAGE_PATH="/opt/amrl/amrl_msgs:$ROS_PACKAGE_PATH"

if [[ "${USE_PY310}" == "1" && -f /opt/py310/bin/activate ]]; then
  # Activate 3.10 venv AFTER sourcing ROS. Do not change system python symlink.
  . /opt/py310/bin/activate
fi

# --- Start roscore if not already running ---
if ! pgrep -x roscore > /dev/null; then
  echo "[entrypoint] Starting roscore in background..."
  roscore &
fi

exec "$@"
