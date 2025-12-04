# ------------------- ProcessImageFunctions.py
import os
import cv2

def converter_pretoebranco(Imagem):
    if not Imagem.valida:
        print("Imagem Invalida")
        return

    Imagem.matriz_NumPy = cv2.cvtColor(Imagem.matriz_NumPy, cv2.COLOR_BGR2GRAY)
    return Imagem


def contar_pixel_hsv(Imagem, hsv_min, hsv_max):
    if not Imagem.valida:
        print("Imagem Invalida")
        return

    imagem_hsv = cv2.cvtColor(Imagem.matriz_NumPy, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(imagem_hsv, hsv_min, hsv_max)
    mascara_pixels = cv2.countNonZero(mascara)
    porcentagem = 100 * mascara_pixels / Imagem.total_pixels

    return {
        "pixels": mascara_pixels,
        "percentual": porcentagem,
        "mascara": mascara
    }
