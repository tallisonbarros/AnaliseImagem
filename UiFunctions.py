# UiFunctions.py
import cv2
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

