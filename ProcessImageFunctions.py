# ------------------- ProcessImageFunctions.py
import os
import cv2
import random
import ImageFunctions
import tkinter.messagebox as mb

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

def analisar(img, valoresHsv):
    if not img.valida:
        return {"erro": "imagem invalida"}

    resultados = {}

    for categoria, hsv_dict in valoresHsv.items():

        hsv_min = hsv_dict["min"]
        hsv_max = hsv_dict["max"]

        info = contar_pixel_hsv(img, hsv_min, hsv_max)

        resultados[categoria] = info["percentual"]

    return resultados

