# Forgeryscope detector image, used by ELIES as a copy-move detection tool.
#
# Build from the repository root (BuildKit is required for the git sources):
#   docker build -f docker/Dockerfile -t forgeryscope:latest .
# GPU build (CUDA PyTorch wheels):
#   docker build -f docker/Dockerfile -t forgeryscope:latest \
#       --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 .
#
# All model weights are downloaded at build time: the container needs no
# network access when it runs. See docker/README.md for usage.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG LIGHTGLUE_COMMIT=eb42fee2d71449efb0aa5c10549752b5d75384d8
ARG ALIKED_COMMIT=683d7c65197395c0b3f01ebe76e1084a27e73a65
ARG FORGERYSCOPE_MODEL_BASE_URL=https://github.com/vlad3996/forgeryscope/releases/download/models-v1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORGERYSCOPE_CACHE_DIR=/opt/forgeryscope/models \
    TORCH_HOME=/opt/forgeryscope/torch

RUN pip install --index-url ${TORCH_INDEX_URL} torch==2.11.0 torchvision==0.26.0

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && pip uninstall -y opencv-python opencv-python-headless \
 && pip install --no-deps opencv-python-headless==4.13.0.92 \
 && rm /tmp/requirements.txt

ADD https://github.com/cvg/LightGlue.git#${LIGHTGLUE_COMMIT} /tmp/LightGlue
RUN pip install --no-deps /tmp/LightGlue && rm -rf /tmp/LightGlue

ADD https://github.com/Shiaoming/ALIKED.git#${ALIKED_COMMIT}:models /tmp/aliked-models
COPY forgeryscope/model_zoo.py docker/fetch_weights.py /tmp/fetch/
RUN mkdir -p ${TORCH_HOME}/hub/checkpoints \
 && cp /tmp/aliked-models/aliked-n16.pth ${TORCH_HOME}/hub/checkpoints/ \
 && FORGERYSCOPE_MODEL_BASE_URL=${FORGERYSCOPE_MODEL_BASE_URL} python /tmp/fetch/fetch_weights.py \
 && rm -rf /tmp/aliked-models /tmp/fetch \
 && chmod -R a+rX /opt/forgeryscope

COPY pyproject.toml README.md LICENSE /tmp/src/
COPY forgeryscope /tmp/src/forgeryscope
RUN pip install --no-deps /tmp/src && rm -rf /tmp/src

COPY docker/run_detection.py /opt/forgeryscope/run_detection.py

# Run offline, and as any user (ELIES' DOCKER_TOOL_USER) with read-only
# model files: everything written at run time goes to /tmp. USER is set
# because PyTorch looks up the user name, which fails for a UID with no
# passwd entry.
ENV YOLO_OFFLINE=true \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HOME=/tmp \
    USER=forgeryscope \
    YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

WORKDIR /tmp
ENTRYPOINT ["python", "/opt/forgeryscope/run_detection.py"]
