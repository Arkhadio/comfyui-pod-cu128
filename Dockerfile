# ComfyUI para pod interactivo de RunPod — Ada / plataforma-ia
# VARIANTE cu128 (repo de respaldo): base CUDA 12.8 — para hosts con driver 12.8-12.x
# donde la imagen CUDA 13 no puede arrancar. torch cu128 incluye kernels Blackwell (sm_120).
# ComfyUI y custom nodes van DENTRO de la imagen (disco local, arranque rápido).
# Los MODELOS viven en el network volume, montado en /workspace.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# ---------- Sistema ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3-pip \
        git wget curl ffmpeg nano \
        libgl1 libglib2.0-0 \
        openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# ---------- ⚠ cuDNN: eliminar el de la imagen base ----------
# La base trae cuDNN 9.8.0, pero PyTorch necesita 9.10.2 (el que instala por pip).
# Si conviven, ReActor peta con CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
RUN rm -f /usr/lib/x86_64-linux-gnu/libcudnn*.so* && ldconfig

# ---------- ⚠ Forward compatibility: eliminarla ----------
# La base trae /usr/local/cuda/compat (libcuda "de repuesto" para hosts con driver
# viejo). Solo funciona en GPUs de datacenter; en la RTX Pro 6000 provoca
# "Error 804: forward compatibility was attempted on non supported HW".
# La quitamos: el contenedor usa siempre el libcuda real del host.
# (Requisito real: desplegar en hosts con driver CUDA >= 12.8 — filtro en RunPod.)
RUN rm -rf /usr/local/cuda/compat /usr/local/cuda-*/compat /etc/ld.so.conf.d/*compat*.conf && ldconfig

# ---------- PyTorch (cu128) ----------
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# ---------- Compilador + llama-cpp (Searge LLM / prompts en espanol) ----------
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    CMAKE_ARGS="-DGGML_NATIVE=OFF" CC=gcc CXX=g++ pip install --no-binary llama-cpp-python llama-cpp-python

# Priorizar el cuDNN que trae PyTorch
RUN echo /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib > /etc/ld.so.conf.d/zz-cudnn.conf && ldconfig

# ---------- ComfyUI ----------
WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /opt/ComfyUI
RUN pip install -r requirements.txt

# ---------- Custom nodes ----------
WORKDIR /opt/ComfyUI/custom_nodes

RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git && \
    pip install -r ComfyUI-Manager/requirements.txt

RUN git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git && \
    pip install -r rgthree-comfy/requirements.txt

RUN git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor.git && \
    pip install -r ComfyUI-ReActor/requirements.txt

# --- ARREGLO REACTOR DEFINITIVO: Desactiva el filtro NSFW forzando el condicional a True ---
RUN sed -i 's/if not sfw.nsfw_image(img_byte_arr, NSFWDET_MODEL_PATH):/if True:/g' /opt/ComfyUI/custom_nodes/ComfyUI-ReActor/nodes.py

RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git comfyui-impact-pack && \
    pip install -r comfyui-impact-pack/requirements.txt

RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git comfyui-impact-subpack && \
    pip install -r comfyui-impact-subpack/requirements.txt

RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git comfyui-videohelpersuite && \
    pip install -r comfyui-videohelpersuite/requirements.txt

RUN git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git comfyui-frame-interpolation && \
    pip install -r comfyui-frame-interpolation/requirements-no-cupy.txt

RUN git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip install -r ComfyUI-KJNodes/requirements.txt

# DWPose para Wan 2.2 Animate. Su requirements incluye onnxruntime: se filtra.
RUN git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    grep -viE "^(onnxruntime|torch|numpy|opencv)" comfyui_controlnet_aux/requirements.txt > /tmp/cnaux_req.txt && \
    pip install -r /tmp/cnaux_req.txt

# SAM2 para Wan Animate modo replacement (no tiene requirements.txt)
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-segment-anything-2.git
RUN test -f "/opt/ComfyUI/custom_nodes/ComfyUI-segment-anything-2/sam2_configs/sam2.1_hiera_b+.yaml" || \
    (mkdir -p /opt/ComfyUI/custom_nodes/ComfyUI-segment-anything-2/sam2_configs && \
     wget -q -O "/opt/ComfyUI/custom_nodes/ComfyUI-segment-anything-2/sam2_configs/sam2.1_hiera_b+.yaml" \
     "https://raw.githubusercontent.com/kijai/ComfyUI-segment-anything-2/main/sam2_configs/sam2.1_hiera_b%2B.yaml")

RUN git clone --depth 1 https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git comfyui-inpaint-cropandstitch

RUN mkdir -p /opt/ComfyUI/models/upscale_models && \
    wget -q -O /opt/ComfyUI/models/upscale_models/4x-UltraSharp.pth \
    "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth"

# ---------- Jupyter Lab con terminal ----------
RUN pip install jupyterlab jupyter-server-terminals packaging

# ---------- ⚠ LÍNEA CRÍTICA: va la ÚLTIMA a propósito ----------
# Si algún custom node ha degradado onnxruntime, esto lo restaura.
# 1.20.1 es la última versión que enlaza contra CUDA 12 (las siguientes piden libcudart.so.13).
RUN pip uninstall -y onnxruntime onnxruntime-gpu || true && \
    pip install --force-reinstall --no-cache-dir "onnxruntime-gpu<1.23" "protobuf<5" "numpy<2.5"

# ---------- Configuración ----------
COPY extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml
COPY start.sh /start.sh
RUN chmod +x /start.sh

WORKDIR /opt/ComfyUI
EXPOSE 8188 8888 22
CMD ["/start.sh"]
