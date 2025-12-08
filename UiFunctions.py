# UiFunctions.py
import cv2
import tkinter as tk
from PIL import Image, ImageTk

def exibir_imagem_ui(imagem_numpy, label_destino):
    """
    Recebe a matriz NumPy da imagem (BGR) e um tk.Label onde
    a imagem será exibida.
    """
    # converter BGR → RGB
    img_rgb = cv2.cvtColor(imagem_numpy, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.resize((350, 350))

    imagem_tk = ImageTk.PhotoImage(pil_img)

    # guardar referência na própria label para não perder a imagem
    label_destino.image = imagem_tk
    label_destino.config(image=imagem_tk)


def criar_inputs_hsv(frame, titulo):
    bloco = tk.LabelFrame(frame, text=titulo)
    bloco.pack(pady=5)

    inputs = {}

    for componente in ["H", "S", "V"]:
        linha = tk.Frame(bloco)
        linha.pack()

        tk.Label(linha, text=componente + "_min").pack(side="left")
        entry_min = tk.Entry(linha, width=5)
        entry_min.pack(side="left")

        tk.Label(linha, text=componente + "_max").pack(side="left")
        entry_max = tk.Entry(linha, width=5)
        entry_max.pack(side="left")

        inputs[componente] = (entry_min, entry_max)

    return inputs

def criar_input_preprocess(frame_pai):
    """
    Cria o bloco completo de pré-processamento contendo:
    - HSV Gérmen
    - HSV Casca
    - HSV Canjica
    Retorna um dicionário com tudo organizado.
    """

    bloco = tk.LabelFrame(frame_pai, text="Pré-Processamento", font=("Arial", 10, "bold"))
    bloco.pack(side="right", padx=10, pady=10)

    # Criar os 3 conjuntos HSV
    hsv_germen  = criar_inputs_hsv(bloco, "Cor Gérmen")
    hsv_casca   = criar_inputs_hsv(bloco, "Cor Casca")
    hsv_canjica = criar_inputs_hsv(bloco, "Cor Canjica")

    # Organizar tudo em um pacote
    preprocess = {
        "germen": hsv_germen,
        "casca": hsv_casca,
        "canjica": hsv_canjica
    }

    return preprocess




def ler_inputs_hsv(campos):
    Hmin = int(campos["H"][0].get())
    Hmax = int(campos["H"][1].get())

    Smin = int(campos["S"][0].get())
    Smax = int(campos["S"][1].get())

    Vmin = int(campos["V"][0].get())
    Vmax = int(campos["V"][1].get())

    return (Hmin, Smin, Vmin), (Hmax, Smax, Vmax)
