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



def analisar(self):
    # Só pra testar: gera três números aleatórios que somam 100
    img = ImageFunctions.Imagem(self.imagem_atual)
    hsv = {}
    for categoria, campos in self.preprocess_inputs.items():
        hsv[categoria] = {}

        for comp, (entry_min, entry_max) in campos.items():
            try:
                    vmin = int(entry_min.get())
                    vmax = int(entry_max.get())
            except:
                    vmin, vmax = 0, 255

            hsv[categoria][comp] = (vmin, vmax)

    mb.showinfo("HSV COLETADO", hsv)       
