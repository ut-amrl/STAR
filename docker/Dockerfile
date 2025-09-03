# ===== Base: CUDA runtime on Ubuntu 20.04 (compatible with ROS Noetic) =====
ARG CUDA_VERSION=12.9.0
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu20.04

# Avoid Python writing .pyc, keep signals, noninteractive apt
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# ---- OS basics + locales ----
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release locales tzdata \
      build-essential git python3 python3-pip python3-venv python3-dev \
      python-is-python3 \
      cmake python3-empy dirmngr && \
    locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# ---- ROS Noetic apt repository + install ----
RUN set -e; \
    apt-get update -y; \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -; \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros1-latest.list; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends \
      ros-noetic-ros-base \
      ros-noetic-tf ros-noetic-tf2-ros ros-noetic-roslint \
      python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool \
      ros-noetic-catkin \
      ros-noetic-rosbuild && \
    rosdep init && rosdep update && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Python userland (minimal) ----
RUN python3 -m pip install --upgrade --no-cache-dir pip wheel setuptools

# ---- amrl_msgs (rosbuild) ----
WORKDIR /opt/amrl
RUN set -e; \
    . /opt/ros/noetic/setup.sh; \
    git clone https://github.com/ut-amrl/amrl_msgs.git; \
    cd amrl_msgs; \
    git checkout taijing/moma/perception; \
    export ROS_PACKAGE_PATH="$(pwd):$ROS_PACKAGE_PATH"; \
    make
# Keep it on PATH for runtime too
ENV ROS_PACKAGE_PATH=/opt/amrl/amrl_msgs:$ROS_PACKAGE_PATH

# Install Python 3.10 side-by-side (keep system python3 = 3.8 for ROS)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/bin/python3.10 -m venv --system-site-packages /opt/py310 && \
    /opt/py310/bin/pip install --upgrade pip wheel setuptools
RUN /opt/py310/bin/pip install --no-cache-dir \
      pymilvus \
      langchain-milvus \
      langchain-huggingface \
      langchain-openai \
      Pillow ipython opencv-python OpenEXR Imath

RUN /opt/py310/bin/pip install --no-cache-dir langchain langgraph langchain-community sentence-transformers

WORKDIR /
RUN printf '%s\n' \
  '[ -f /opt/ros/noetic/setup.bash ] && . /opt/ros/noetic/setup.bash' \
  'export ROS_PACKAGE_PATH=/opt/amrl/amrl_msgs:$ROS_PACKAGE_PATH' \
  'if [ "$USE_PY310" = "1" ] && [ -f /opt/py310/bin/activate ]; then' \
  '  . /opt/py310/bin/activate' \
  'fi' \
  > /etc/profile.d/ros_py310.sh

# ---- Final cleanup ----
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    python3 -m pip cache purge

# ---- ROS environment setup ----
COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
