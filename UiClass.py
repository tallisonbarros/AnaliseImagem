# ------------------- UiClass.py
import tkinter as tk
import tkinter.messagebox as mb
import FileFunctions
import ImageFunctions
import ProcessImageFunctions
import UiFunctions
import os
import cv2
import numpy as np
from PIL import Image


class ClassificadorUI:
    def __init__(self, root, base_dir):
        self.root = root
        self.dirDataBase = os.path.join(base_dir, "ImgDataBase")
        self.img = None
        self.pickcolor = None

        self.root.geometry("1100x800")
        self.root.eval('tk::PlaceWindow . center')

        for c in range(3):
            self.root.columnconfigure(c, weight=1)

        # Header
        header = tk.Frame(self.root)
        header.grid(row=0, column=0, columnspan=3, pady=10)

        # Coluna 1 - imagem + botoes
        col1 = tk.Frame(self.root)
        col1.grid(row=1, column=0, sticky="n")

        self.canvas = tk.Canvas(col1, width=380, height=380, bg="black")
        self.canvas.pack(pady=5)

        self.navigator = UiFunctions.CanvasNavigator(self.canvas)

        # 👉 Volta a usar BOTÃO DIREITO para pegar cor
        self.canvas.bind("<Button-3>", self.pegar_cor, add="+")

        self.TxtDirImagem = tk.Label(col1, font=("Arial", 12))
        self.TxtDirImagem.pack()
        self.TxtQtdItens = tk.Label(col1, font=("Arial", 9))
        self.TxtQtdItens.pack()

        frame_classes = tk.Frame(col1)
        frame_classes.pack(pady=8)
        for i in range(1, 11):
            tk.Button(
                frame_classes,
                text=str(i),
                width=4,
                command=lambda n=i: self.classificar_imagem(n)
            ).grid(row=0, column=i)

        tk.Button(col1, text="Analisar", command=self.analisar).pack(pady=5)
        tk.Button(col1, text="Desclassificar Tudo", command=self.desclassificar_tudo).pack()

        # Coluna 2 - paletas
        col2 = tk.Frame(self.root)
        col2.grid(row=1, column=1, sticky="n")

        tk.Label(col2, text="Pré-Processamento", font=("Arial", 11, "bold")).pack()

        self.paletas = UiFunctions.PaletaCategoriaCores(
            self.get_pickcolor,
            self.on_paleta_change
        )
        self.paletas.criar_paletas(col2)

        # Coluna 3 - análise
        col3 = tk.Frame(self.root)
        col3.grid(row=1, column=2, sticky="n")
        tk.Label(col3, text="Análise automática", font=("Arial", 11, "bold")).pack()

        self.lbl_g = tk.Label(col3, text="Gérmen:  -- %")
        self.lbl_c = tk.Label(col3, text="Casca:   -- %")
        self.lbl_k = tk.Label(col3, text="Canjica: -- %")

        self.lbl_g.pack(anchor="w")
        self.lbl_c.pack(anchor="w")
        self.lbl_k.pack(anchor="w")

        self.root.after(300, self.buscar_imagem)

    def buscar_imagem(self):
        pasta = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
        pasta.filtrar_arquivos(".png")

        if not pasta.lista_arquivos:
            self.img = None
            self.canvas.delete("all")
            self.TxtDirImagem.config(text="")
            self.TxtQtdItens.config(text="")
            mb.showinfo("Fim", "Nenhuma imagem não classificada.")
            return

        caminho = pasta.lista_arquivos[0]
        self.img = ImageFunctions.Imagem(caminho)

        rgb = cv2.cvtColor(self.img.matriz_NumPy, cv2.COLOR_BGR2RGB)
        self.navigator.set_image(Image.fromarray(rgb))

        self.TxtDirImagem.config(text=self.img.nome)
        self.TxtQtdItens.config(text=f"{pasta.quantidade_arquivos} imagens para classificar.")

    def classificar_imagem(self, classe):
        if not self.img:
            return
        pasta = FileFunctions.Pasta(self.dirDataBase, str(classe))
        if FileFunctions.MoverArquivo(self.img.caminho, pasta.dir).get("ok"):
            self.buscar_imagem()

    def desclassificar_tudo(self):
        pasta_nc = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
        destino = pasta_nc.dir

        total = 0
        for i in range(1, 11):
            pasta = FileFunctions.Pasta(self.dirDataBase, str(i))
            for arq in pasta.lista_arquivos:
                if FileFunctions.MoverArquivo(arq, destino).get("ok"):
                    total += 1

        mb.showinfo("OK", f"{total} imagens movidas.")
        self.buscar_imagem()

    def pegar_cor(self, event):
        if not self.img:
            return

        coords = self.navigator.canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return

        x, y = coords
        b, g, r = self.img.matriz_NumPy[y, x]

        hsv = cv2.cvtColor(
            np.uint8([[[b, g, r]]]),
            cv2.COLOR_BGR2HSV
        )[0][0]

        self.pickcolor = hsv

        H, S, V = hsv
        self.paletas.HSVcor.config(bg=UiFunctions.rgb_from_hsv_hex(H, S, V))

    def on_paleta_change(self):
        pass

    def get_pickcolor(self):
        return self.pickcolor

    def analisar(self):
        if not self.img:
            return

        tolerancia = 0.10
        faixas = ProcessImageFunctions.gerar_faixas_hsv(self.paletas.refs, tolerancia)
        r = ProcessImageFunctions.processar_valores_hsv(self.img, faixas)

        self.lbl_g.config(text=f"Gérmen:  {r.get('germen', 0):.1f}%")
        self.lbl_c.config(text=f"Casca:   {r.get('casca', 0):.1f}%")
        self.lbl_k.config(text=f"Canjica: {r.get('canjica', 0):.1f}%")
