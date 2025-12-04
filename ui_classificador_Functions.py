# ui_classificador_Functions.py
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import FileFunctions
import ImageFunctions
import UiFunctions

def buscar_imagem_not_class(dirDataBase, label_destino):

    # monta a pasta ImgNotClass dentro do database
    pasta_nc = FileFunctions.Pasta(dirDataBase, "ImgNotClass")

    # filtra apenas PNG
    pasta_nc.filtrar_arquivos(".png")

    if not pasta_nc.lista_arquivos:
        messagebox.showinfo("Fim", "Nenhuma imagem não classificada encontrada.")
        return None

    caminho = pasta_nc.lista_arquivos[0]

    # monta o objeto imagem
    img = ImageFunctions.Imagem(caminho)
    if not img.valida:
        messagebox.showerror("Imagem invalida", img.erro)
        return None

    # exibe na UI
    UiFunctions.exibir_imagem_ui(img.matriz_NumPy, label_destino)

    return caminho

