# ------------------- cnn_quebra_clf.py
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
_DEVICE = None
_MODEL_PATH_DEFAULT = os.getenv("CNN_QUEBRA_PTH", os.path.join(os.getcwd(), "cnn_quebra.pt"))
_INPUT_SIZE = int(os.getenv("CNN_QUEBRA_SIZE", "224") or "224")


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


def classificar_quebra(img_bgr, mask_fg=None, model_path=None):
    """
    Classifica imagem inteira: 0=integro, 1=parcial, 2=quebrado.
    Retorna dict {classe_id, classe_nome, probs(list)}.
    Se modelo/torch indisponivel, retorna None.
    """
    if img_bgr is None or not HAS_TORCH:
        return None

    modelo = _carregar_modelo(model_path=model_path)
    if modelo is None:
        return None

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    mask = mask_fg if mask_fg is not None else np.ones((h, w), dtype=np.uint8) * 255
    mask_bool = mask.astype(bool)
    if mask_bool.any():
        bgr = img_bgr.copy()
        bgr[~mask_bool] = (0, 0, 0)
    else:
        bgr = img_bgr

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    alvo = max(1, _INPUT_SIZE)
    if alvo != h or alvo != w:
        tensor_img = F.interpolate(tensor_img, size=(alvo, alvo), mode="bilinear", align_corners=False)

    device = _DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        logits = modelo(tensor_img)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    probs_np = probs.detach().cpu().numpy().tolist()
    cls_id = int(np.argmax(probs_np))
    cls_nome = ["integro", "parcial", "quebrado"][cls_id]

    return {
        "classe_id": cls_id,
        "classe_nome": cls_nome,
        "probs": probs_np,
    }
