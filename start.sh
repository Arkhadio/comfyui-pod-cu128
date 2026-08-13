#!/bin/bash
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib

echo "=== Pod ComfyUI — arranque (variante cu128) ==="

# ---------- DNS ----------
# El resolvedor interno de Docker falla en algunos hosts de RunPod.
if ! python -c "import socket; socket.gethostbyname('github.com')" 2>/dev/null; then
    echo "DNS roto, usando 8.8.8.8"
    echo "nameserver 8.8.8.8" > /etc/resolv.conf
fi 

# ---------- Actualizar ComfyUI (Animate2 / SCAIL / nodos nuevos) ----------
echo "Actualizando ComfyUI..."
( cd /opt/ComfyUI && git pull -q && \
  pip install --break-system-packages -q -r requirements.txt ) \
  && echo "  ComfyUI actualizado" || echo "  aviso: sin actualizar (sin red o conflicto)"

# ---------- SSH ----------
if [[ -n "$PUBLIC_KEY" ]]; then
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    service ssh start || true
fi

# ---------- Carpetas persistentes ----------
mkdir -p /workspace/comfy/output /workspace/comfy/user /opt/ComfyUI/input
# ---------- Esqueleto del volumen (para volúmenes nuevos/vacíos) ----------
VOLBASE=/workspace/runpod-slim/ComfyUI
mkdir -p "$VOLBASE/custom_nodes"
for d in checkpoints clip clip_vision configs controlnet diffusion_models \
         embeddings loras text_encoders ultralytics/bbox ultralytics/segm \
        upscale_models vae sams reactor insightface model_patches ipadapter detection llm_gguf; do
    mkdir -p "$VOLBASE/models/$d"
done

# ---------- Enlaces a modelos que NO respetan extra_model_paths.yaml ----------
# ReActor y sus dependencias buscan en rutas fijas dentro de /opt/ComfyUI/models.
VOL=/workspace/runpod-slim/ComfyUI/models
link_model_dir() {
    local nombre=$1
    if [[ -d "$VOL/$nombre" ]]; then
        rm -rf "/opt/ComfyUI/models/$nombre"
        ln -sfn "$VOL/$nombre" "/opt/ComfyUI/models/$nombre"
        echo "  enlazado: $nombre"
    fi
}

echo "Enlazando modelos del volumen..."
mkdir -p /opt/ComfyUI/models
link_model_dir reactor
link_model_dir hyperswap
link_model_dir insightface
link_model_dir facedetection
link_model_dir facerestore_models
link_model_dir llm_gguf
link_model_dir detection
link_model_dir face_parsing

# ---------- Custom nodes del volumen (persistentes) ----------
# Los custom nodes que se instalan despues (via Manager o git) viven en el volumen.
# ComfyUI solo mira /opt/ComfyUI/custom_nodes, asi que los enlazamos en cada arranque.
VOL_NODES=/workspace/runpod-slim/ComfyUI/custom_nodes
if [[ -d "$VOL_NODES" ]]; then
    echo "Enlazando custom nodes del volumen..."
    for d in "$VOL_NODES"/*/; do
        nombre=$(basename "$d")
        destino="/opt/ComfyUI/custom_nodes/$nombre"
        # si la imagen trae una copia real, se elimina para que mande el volumen
        if [[ -d "$destino" && ! -L "$destino" ]]; then
            rm -rf "$destino"
        fi
        ln -sfn "${d%/}" "$destino" && echo "  enlazado nodo: $nombre"
    done
fi

# ---------- Librerias pip que usan los custom nodes del volumen ----------
# No estan en la imagen base; se reinstalan aqui (rapido si ya estan en cache del volumen).
echo "Verificando librerias pip..."
pip install --break-system-packages -q \
    insightface timm mediapipe==0.10.14 blend_modes facexlib kornia accelerate \
    gguf qwen_vl_utils sageattention onnxruntime-gpu \
    2>/dev/null && echo "  librerias pip OK"
pip install --break-system-packages -q -U diffusers 2>/dev/null && echo "  diffusers actualizado"

# ---------- clip_vision con nombre alternativo (Wan Animate) ----------
CV=/workspace/runpod-slim/ComfyUI/models/clip_vision
if [[ -f "$CV/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" ]]; then
    ln -sf "$CV/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" "$CV/clip_vision_h.safetensors" 2>/dev/null
fi

# ---------- Jupyter Lab ----------
if [[ "${START_JUPYTER:-1}" == "1" ]]; then
    nohup jupyter lab --allow-root --no-browser --port=8888 --ip=0.0.0.0 \
        --ServerApp.token='' --ServerApp.password='' \
        --ServerApp.terminals_enabled=True \
        --ServerApp.allow_origin='*' \
        --ServerApp.allow_remote_access=True \
        --ServerApp.disable_check_xsrf=True \
        --notebook-dir=/workspace > /var/log/jupyter.log 2>&1 &
    echo "Jupyter Lab en el puerto 8888 (con terminal)"
fi

# ---------- Comprobaciones ----------
echo "onnxruntime: $(python -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null || echo 'NO DISPONIBLE')"
echo "cuDNN: $(python -c 'import torch; print(torch.backends.cudnn.version())' 2>/dev/null || echo 'NO DISPONIBLE')"

# ---------- Chequeo de GPU / driver del host ----------
# Si el host no puede inicializar CUDA (driver demasiado viejo, error 804, etc.)
# NO arrancamos ComfyUI en bucle: dejamos Jupyter vivo y un mensaje claro en el log.
echo "Driver del host: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi no disponible')"
if ! python - <<'PY'
import sys
try:
    import torch
    torch.cuda.init()
    print(f"GPU OK: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"FALLO CUDA: {e}", file=sys.stderr)
    sys.exit(1)
PY
then
    echo "============================================================"
    echo " ✖ GPU NO UTILIZABLE EN ESTE HOST"
    echo "   El driver de esta maquina no soporta CUDA 12.8 (o CUDA no"
    echo "   inicializa). ComfyUI NO se arranca para evitar el bucle."
    echo "   SOLUCION: Terminate del pod y redeploy eligiendo en"
    echo "   'Additional filters' -> CUDA Version >= 12.8."
    echo "   Jupyter sigue vivo en el puerto 8888 para inspeccionar."
    echo "============================================================"
    sleep infinity
fi

cd /opt/ComfyUI
echo "Arrancando ComfyUI en el puerto 8188..."
exec python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --output-directory /opt/ComfyUI/output \
    --input-directory /opt/ComfyUI/input \
    --user-directory /workspace/comfy/user
