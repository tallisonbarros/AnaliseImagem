# ------------------- cnn_face_interna.py
import os
import threading
import numpy as np
import cv2

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    F = None

# cache do modelo carregado
_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_PATH_DEFAULT = os.getenv("CNN_FACE_INTERNA_PTH", os.path.join(os.getcwd(), "cnn_face_interna.pt"))
_INPUT_SIZE = int(os.getenv("CNN_FACE_INTERNA_SIZE", "256") or "256")
_DEVICE = None


def _carregar_modelo(model_path=None, device=None):
    """Carrega modelo .pt apenas uma vez (jit ou state_dict completo)."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    if not HAS_TORCH:
        return None

    caminho = model_path or _MODEL_PATH_DEFAULT
    if not os.path.isfile(caminho):
        return None

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            modelo = torch.jit.load(caminho, map_location=dev)
        except Exception:
            try:
                modelo = torch.load(caminho, map_location=dev)
            except Exception:
                return None
        modelo.eval()
        _MODEL = modelo.to(dev)
        _DEVICE = dev
    return _MODEL


def gerar_mapas_face_interna_cnn(img_bgr, mask_fg=None, model_path=None):
    """
    Gera mapa de probabilidade de face interna via CNN.
    Retorna dict com mapa_interno/mapa_confianca normalizados em [0,1].
    Se o modelo ou torch nao estiverem disponiveis, devolve None para permitir fallback.
    """
    if img_bgr is None:
        return {"mapa_interno": None, "mapa_confianca": None}
    if not HAS_TORCH:
        return {"mapa_interno": None, "mapa_confianca": None}

    modelo = _carregar_modelo(model_path=model_path)
    if modelo is None:
        return {"mapa_interno": None, "mapa_confianca": None}

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return {"mapa_interno": None, "mapa_confianca": None}

    mask = mask_fg if mask_fg is not None else np.ones((h, w), dtype=np.uint8) * 255
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return {"mapa_interno": None, "mapa_confianca": None}

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb[~mask_bool] = 0.0

    tensor_img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    tensor_mask = torch.from_numpy(mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    alvo = max(1, _INPUT_SIZE)
    if alvo != h or alvo != w:
        tensor_img = F.interpolate(tensor_img, size=(alvo, alvo), mode="bilinear", align_corners=False)
        tensor_mask = F.interpolate(tensor_mask, size=(alvo, alvo), mode="nearest")
    tensor_img = tensor_img * tensor_mask

    device = _DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    tensor_img = tensor_img.to(device)
    tensor_mask = tensor_mask.to(device)

    with torch.no_grad():
        saida = modelo(tensor_img)

    if isinstance(saida, dict):
        for val in saida.values():
            saida = val
            break
    if isinstance(saida, (list, tuple)):
        saida = saida[0]
    if not torch.is_tensor(saida):
        return {"mapa_interno": None, "mapa_confianca": None}
    if saida.dim() == 3:
        saida = saida.unsqueeze(1)
    if saida.dim() != 4:
        return {"mapa_interno": None, "mapa_confianca": None}

    mapa = torch.sigmoid(saida[:, 0:1])
    mapa = F.interpolate(mapa, size=(h, w), mode="bilinear", align_corners=False)
    mapa_np = mapa.squeeze(0).squeeze(0).detach().cpu().numpy()
    mapa_np = np.clip(mapa_np, 0.0, 1.0)
    mapa_np = mapa_np * mask_bool.astype(np.float32)

    mapa_confianca = (mapa_np > 0).astype(np.float32) * mask_bool.astype(np.float32)

    return {
        "mapa_interno": mapa_np,
        "mapa_confianca": mapa_confianca,
    }
