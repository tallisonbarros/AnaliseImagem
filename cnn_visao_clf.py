# ------------------- cnn_visao_clf.py
"""
Inferencia do classificador de visao (notas 1..10) usando modelo TorchScript.
"""
import os
import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None

MODEL_PATH = os.getenv("CNN_VISAO_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_visao.pt"))
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not HAS_TORCH:
        raise RuntimeError("PyTorch nao instalado.")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Modelo de visao nao encontrado: {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL = torch.jit.load(MODEL_PATH, map_location=device)
    _MODEL.eval()
    return _MODEL


def classificar_visao(img_bgr, tamanho=224):
    """
    Recebe imagem BGR e retorna dict com probs (len=10) e indice (1-10).
    """
    modelo = _load_model()
    device = next(modelo.parameters()).device if hasattr(modelo, "parameters") else "cpu"

    bgr = cv2.resize(img_bgr, (tamanho, tamanho), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = modelo(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pesos = np.arange(1, len(probs) + 1, dtype=np.float32)
    soma_probs = float(np.sum(probs))
    indice = float(np.dot(pesos, probs) / soma_probs) if soma_probs > 0 else float(np.argmax(probs) + 1)
    classe = int(round(indice))
    return {"probs": probs.tolist(), "indice_visao": indice, "classe": str(classe)}
