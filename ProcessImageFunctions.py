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

def processar_valores_hsv(img, valoresHsv):
    if not img.valida:
        return {"erro": "imagem invalida"}

    resultados = {}

    total_pixels = img.qtd_pixels()   # você deve ter essa função; senão eu te passo

    for categoria, lista_faixas in valoresHsv.items():

        soma_pixels_categoria = 0

        # percorre cada faixa cadastrada
        for faixa in lista_faixas:
            hsv_min = faixa["min"]
            hsv_max = faixa["max"]

            info = contar_pixel_hsv(img, hsv_min, hsv_max)

            soma_pixels_categoria += info["pixels"]

        # converte para percentual
        percentual = (soma_pixels_categoria / total_pixels) * 100.0

        resultados[categoria] = percentual

    return resultados

