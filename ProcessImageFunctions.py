# ------------------- ProcessImageFunctions.py
import cv2
import numpy as np

def contar_pixel_hsv(Imagem, hsv_min, hsv_max):
    """Conta quantos pixels estão entre min/max usando HSV já pré-calculado."""
    mascara = cv2.inRange(Imagem.hsv, hsv_min, hsv_max)
    pixels = cv2.countNonZero(mascara)
    return pixels


def gerar_faixas_hsv(paletas, tolerancia=0.10):
    """
    Gera faixas HSV com tolerância proporcional.
    Exemplo: H=50 com tolerância 10% => faixa [45 ... 55]
    """
    faixas = {}

    for categoria, dados in paletas.items():
        if categoria == "HSVCor":
            continue

        if "cores" not in dados:
            continue

        cores = dados["cores"]
        if not cores:
            continue

        lista_faixas = []

        for (H, S, V) in cores:
            # deltas proporcionais
            h_delta = int(H * tolerancia)
            s_delta = int(S * tolerancia)
            v_delta = int(V * tolerancia)

            h_min = max(H - h_delta, 0)
            h_max = min(H + h_delta, 179)

            s_min = max(S - s_delta, 0)
            s_max = min(S + s_delta, 255)

            v_min = max(V - v_delta, 0)
            v_max = min(V + v_delta, 255)

            lista_faixas.append({
                "min": (h_min, s_min, v_min),
                "max": (h_max, s_max, v_max)
            })

        faixas[categoria] = lista_faixas

    return faixas


def processar_valores_hsv(img, faixas):
    """Processa valores HSV usando faixas geradas com tolerância."""
    resultados = {}
    total = img.total_pixels

    for categoria, lista_faixas in faixas.items():
        soma = 0
        for faixa in lista_faixas:
            soma += contar_pixel_hsv(img, faixa["min"], faixa["max"])

        resultados[categoria] = (soma / total) * 100.0

    return resultados
