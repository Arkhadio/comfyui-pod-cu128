# comfyui-pod

Imagen Docker para el pod interactivo de RunPod (proyecto Ada / plataforma-ia).

## Qué resuelve

Antes: ComfyUI vivía en el network volume. Cada migración del pod borraba los paquetes de
pip (`onnxruntime`, dependencias de custom nodes) y el arranque tardaba ~20 minutos, porque
miles de archivos `.py` se leían por FUSE.

Ahora: **ComfyUI y los custom nodes van dentro de la imagen** (disco local, arranque en
segundos). En el volumen quedan solo los **modelos**, conectados con `extra_model_paths.yaml`.

## Qué incluye

- Ubuntu 24.04 + CUDA 12.8 + Python 3.12
- PyTorch 2.10.0+cu128
- ComfyUI
- Custom nodes: ComfyUI-Manager, rgthree, ReActor, Impact Pack + Subpack,
  VideoHelperSuite, Frame-Interpolation (RIFE), KJNodes, controlnet_aux (DWPose),
  segment-anything-2 (SAM2)
- Jupyter Lab
- **`onnxruntime-gpu==1.20.1`, instalado en la ÚLTIMA línea del Dockerfile**

### Por qué esa última línea importa

Varios custom nodes traen `onnxruntime` en su `requirements.txt` y lo pisan.
La versión 1.20.1 es la última que enlaza contra CUDA 12; las posteriores piden
`libcudart.so.13` y ReActor deja de funcionar. Al instalarla al final, cualquier degradación
previa queda revertida.

Los `requirements.txt` de controlnet_aux, SAM2 e Impact se filtran para que no toquen
`onnxruntime`, `torch`, `numpy` ni `opencv`.

## Publicación

GitHub Actions construye y publica en `ghcr.io/<usuario>/comfyui-pod:latest` con cada push
a `main`. No hace falta Docker Hub ni Docker en local.

**Haz el paquete público** tras el primer build:
Repo → Packages → comfyui-pod → Package settings → Change visibility → Public.
Si no, RunPod no podrá descargarlo sin credenciales.

## Uso en RunPod

Al crear el pod:

- **Container image:** `ghcr.io/<usuario>/comfyui-pod:latest`
- **Volume mount path:** `/workspace`
- **Expose HTTP ports:** `8188,8888`
- **Container disk:** 20 GB como mínimo

### Rutas

| Qué | Dónde |
|---|---|
| ComfyUI | `/opt/ComfyUI` (imagen, efímero) |
| Modelos | `/workspace/runpod-slim/ComfyUI/models` (volumen) |
| Salidas | `/workspace/comfy/output` (volumen) |
| Entradas | `/workspace/comfy/input` (volumen) |
| Workflows guardados | `/workspace/comfy/user` (volumen) |

Nada que instales dentro del pod persiste. Si necesitas un custom node nuevo,
**añádelo al Dockerfile y haz push**: es la única forma correcta.

## Verificación tras el primer arranque

```bash
python -c "import onnxruntime; print(onnxruntime.__version__, onnxruntime.get_available_providers())"
# → 1.20.1 [... 'CUDAExecutionProvider' ...]
```

Y carga `faceswap_foto_ada_final.json`: si sale la cara de Ada, todo está bien.
