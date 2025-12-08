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


def criar_inputs_hsv(frame, titulo, valores_padrao=None):
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

        # ✔️ Preenche valores padrão, se fornecidos
        if valores_padrao:
            vmin, vmax = valores_padrao.get(componente, (0, 255))
            entry_min.insert(0, str(vmin))
            entry_max.insert(0, str(vmax))

        inputs[componente] = (entry_min, entry_max)

    return inputs

def criar_input_preprocess(frame_pai):

    bloco = tk.LabelFrame(frame_pai, text="Pré-Processamento", font=("Arial", 10, "bold"))
    bloco.pack(side="right", padx=10, pady=10)

    # ===== VALORES PADRÃO =====

    hsv_germen = criar_inputs_hsv(
        bloco, "Germen - Azul",
        valores_padrao={
            "H": (90, 115),     # azul OpenCV
            "S": (200, 255),
            "V": (150, 255)
        }
    )

    hsv_casca = criar_inputs_hsv(
        bloco, "Casca - Amarelo",
        valores_padrao={
            "H": (25, 30),      # amarelo OpenCV
            "S": (200, 255),
            "V": (200, 255)
        }
    )

    hsv_canjica = criar_inputs_hsv(
        bloco, "Canjica - Preto",
        valores_padrao={
            "H": (0, 179),      # preto não tem H, então deixa amplo
            "S": (0, 50),
            "V": (0, 50)
        }
    )

    HSVcor = tk.Label(bloco, text="Cor:", font=("Arial", 8, "bold"))
    HSVcor.pack()

    preprocess = {
        "germen": hsv_germen,
        "casca": hsv_casca,
        "canjica": hsv_canjica,
        "HSVCor": HSVcor
    }
    
    return preprocess

def coletar_hsv(preprocess_inputs):
    resultado = {}

    for categoria, campos in preprocess_inputs.items():
        try:
            Hmin = int(campos["H"][0].get())
            Hmax = int(campos["H"][1].get())
            Smin = int(campos["S"][0].get())
            Smax = int(campos["S"][1].get())
            Vmin = int(campos["V"][0].get())
            Vmax = int(campos["V"][1].get())
        except:
            Hmin, Smin, Vmin = 0, 0, 0
            Hmax, Smax, Vmax = 179, 255, 255

        resultado[categoria] = {
            "min": (Hmin, Smin, Vmin),
            "max": (Hmax, Smax, Vmax)
        }

    return resultado
