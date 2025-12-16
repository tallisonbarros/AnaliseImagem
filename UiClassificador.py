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
from PIL import Image, ImageTk
from datetime import datetime


BG_COLOR = "#5c5c5c"


class ClassificadorUI:
    def __init__(self, root, base_dir):
        self.root = root
        self.dirDataBase = os.path.join(base_dir, "ImgDataBase")
        self.base_dir = base_dir
        self.img = None
        self.pickcolor = None
        self.logo_img = None

        self.root.geometry("1100x800")
        self.root.eval('tk::PlaceWindow . center')
        self.root.configure(bg=BG_COLOR)
        self.root.option_add("*Background", BG_COLOR)

        for c in range(3):
            self.root.columnconfigure(c, weight=1)

        # Header (espacamento)
        self.header = tk.Frame(self.root, bg=BG_COLOR)
        self.header.grid(row=0, column=0, columnspan=3, pady=10)

        # Coluna 1 - imagem + botoes
        col1 = tk.Frame(self.root, bg=BG_COLOR)
        col1.grid(row=1, column=0, sticky="n")

        self.canvas = tk.Canvas(col1, width=250, height=250, bg="black")
        self.canvas.pack(pady=5)

        self.navigator = UiFunctions.CanvasNavigator(self.canvas)

        self.canvas.bind("<Button-3>", self.pegar_cor, add="+")

        self.TxtDirImagem = tk.Label(col1, font=("Arial", 12), bg=BG_COLOR)
        self.TxtDirImagem.pack()
        self.TxtQtdItens = tk.Label(col1, font=("Arial", 9), bg=BG_COLOR)
        self.TxtQtdItens.pack()

        frame_classes = tk.Frame(col1, bg=BG_COLOR)
        frame_classes.pack(pady=8)
        for i in range(1, 11):
            tk.Button(
                frame_classes,
                text=str(i),
                width=4,
                command=lambda n=i: self.classificar_imagem(n)
            ).grid(row=0, column=i)

        tk.Button(col1, text="Analisar Imagem", command=self.analisar_imagem).pack(pady=2)
        tk.Button(col1, text="Analisar Composicao", command=self.analisar_composicao).pack(pady=5)
        tk.Button(col1, text="Analisar Score", command=self.analisar_score).pack(pady=2)
        tk.Button(col1, text="Analisar Quebra", command=self.analisar_quebra).pack(pady=2)
        tk.Button(col1, text="separar componentes", command=self.separar_componentes).pack(pady=2)
        tk.Button(col1, text="Desclassificar Tudo", command=self.desclassificar_tudo).pack()

        # Coluna 2 - paletas
        col2 = tk.Frame(self.root, bg=BG_COLOR)
        col2.grid(row=1, column=1, sticky="n")

        tk.Label(col2, text="Pre-Processamento", font=("Arial", 11, "bold"), bg=BG_COLOR).pack()

        self.paletas = UiFunctions.PaletaMetaDados(
            self.get_pickcolor,
            self.on_paleta_change
        )
        self.paletas.criar_paletas(col2)

        # Coluna 3 - analise
        col3 = tk.Frame(self.root, bg=BG_COLOR)
        col3.grid(row=1, column=2, sticky="n")
        tk.Label(col3, text="Analise", font=("Arial", 11, "bold"), bg=BG_COLOR).pack()

        bloco_cats = tk.LabelFrame(col3, text="Categorias", bg=BG_COLOR, fg="white")
        bloco_cats.pack(fill="both", padx=4, pady=(4, 2))

        self.lbl_g = tk.Label(bloco_cats, text="Germen:   -- %", bg=BG_COLOR)
        self.lbl_c = tk.Label(bloco_cats, text="Casca:    -- %", bg=BG_COLOR)
        self.lbl_k = tk.Label(bloco_cats, text="Canjica:  -- %", bg=BG_COLOR)

        self.lbl_g.pack(anchor="w", padx=6, pady=1)
        self.lbl_c.pack(anchor="w", padx=6, pady=1)
        self.lbl_k.pack(anchor="w", padx=6, pady=1)

        bloco_resumo = tk.LabelFrame(col3, text="Resumo", bg=BG_COLOR, fg="white")
        bloco_resumo.pack(fill="both", padx=4, pady=(2, 4))

        self.lbl_u = tk.Label(bloco_resumo, text="Util:     -- %", bg=BG_COLOR)
        self.lbl_f = tk.Label(bloco_resumo, text="Exclusao: -- %", bg=BG_COLOR)

        self.lbl_u.pack(anchor="w", padx=6, pady=1)
        self.lbl_f.pack(anchor="w", padx=6, pady=1)

        bloco_deger = tk.LabelFrame(col3, text="Degerminação", bg=BG_COLOR, fg="white")
        bloco_deger.pack(fill="both", padx=4, pady=(2, 4))
        
        self.lbl_deger = tk.Label(bloco_deger, text="Score --", bg=BG_COLOR)
        self.lbl_deger.pack(anchor="w", padx=6, pady=1)
        self.lbl_quebra = tk.Label(bloco_deger, text="Quebra: --", bg=BG_COLOR)
        self.lbl_quebra.pack(anchor="w", padx=6, pady=1)

        self.root.after(300, self.buscar_imagem)
        self._carregar_logo()

    def _carregar_logo(self):
        caminho_logo = os.path.join(
            self.base_dir,
            "imagens",
            "logos",
            "LOGO-YRIS.png"
        )
        if not os.path.isfile(caminho_logo):
            return
        try:
            logo = Image.open(caminho_logo)
            # redimensiona para altura fixa de 150 px preservando a proporcao
            w, h = logo.size
            if h > 0:
                nova_altura = 30
                nova_largura = max(1, int(w * (nova_altura / h)))
                logo = logo.resize((nova_largura, nova_altura), Image.LANCZOS)
            self.logo_img = ImageTk.PhotoImage(logo)
            logo_frame = tk.Frame(self.header, bg=BG_COLOR)
            logo_frame.pack()
            tk.Label(logo_frame, image=self.logo_img, bg=BG_COLOR).pack()
        except Exception:
            self.logo_img = None

    def _atualizar_canvas(self):
        if not self.img:
            self.canvas.delete("all")
            return
        rgb = cv2.cvtColor(self.img.matriz_NumPy, cv2.COLOR_BGR2RGB)
        self.navigator.set_image(Image.fromarray(rgb))

    def buscar_imagem(self):
        pasta = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
        pasta.filtrar_arquivos(".png")

        if not pasta.lista_arquivos:
            self.img = None
            self.canvas.delete("all")
            self.TxtDirImagem.config(text="")
            self.TxtQtdItens.config(text="")
            mb.showinfo("Fim", "Nenhuma imagem nao classificada.")
            return

        caminho = pasta.lista_arquivos[0]
        self.img = ImageFunctions.Imagem(caminho)

        self._atualizar_canvas()
        self.analisar_imagem()

        self.TxtDirImagem.config(text=self.img.nome)
        self.TxtQtdItens.config(text=f"{pasta.quantidade_arquivos} imagens para classificar.")
        # pontos do SAM agora sao automáticos (HSV + fundo), nada a fazer aqui

    def classificar_imagem(self, score_val):
        if not self.img:
            return

        # calcula composicao via IA antes de mover/salvar
        comp = ProcessImageFunctions.calcular_composicao(self.img, self.paletas.refs) or {}

        # grava historico em JSON/CSV
        registro = {
            "arquivo": self.img.nome,
            "caminho": self.img.caminho,
            "score": score_val,
            "germen": round(comp.get("germen", 0.0), 4),
            "casca": round(comp.get("casca", 0.0), 4),
            "canjica": round(comp.get("canjica", 0.0), 4),
            "util": round(comp.get("util", 0.0), 4),
            "exclusao": round(comp.get("exclusao", 0.0), 4),
            "data_hora": datetime.now().isoformat(timespec="seconds"),
        }
        #FileFunctions.registrar_score(registro)

        # mover ainda suportado (pode ser desativado futuramente)
        pasta = FileFunctions.Pasta(self.dirDataBase, str(score_val))
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

    def _atualizar_label_sam(self):
        pass

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
        self.paletas.atualizar_cor_preview(UiFunctions.rgb_from_hsv_hex(H, S, V))

    def on_paleta_change(self):
        # Sempre que a paleta muda, resetamos os percentuais ate a proxima analise
        self.lbl_g.config(text="Germen:   -- %")
        self.lbl_c.config(text="Casca:    -- %")
        self.lbl_k.config(text="Canjica:  -- %")
        self.lbl_u.config(text="Util:     -- %")
        self.lbl_f.config(text="Exclusao: -- %")
        self.lbl_deger.config(text="Score --")
        self.lbl_quebra.config(text="Quebra: --")

    def get_pickcolor(self):
        return self.pickcolor

    def separar_componentes(self):
        if not self.img:
            return

        pasta_destino = os.path.join(self.dirDataBase, "ImgNotClass")
        salvos = ProcessImageFunctions.separar_componentes_sam(self.img, pasta_destino, paletas=self.paletas.refs if self.paletas else None)

        if not salvos:
            mb.showinfo("Separar componentes", "Nenhuma camada salva.")
            return

        mb.showinfo(
            "Separar componentes",
            f"{len(salvos)} componente(s) salvo(s) em ImgNotClass."
        )

    def analisar_imagem(self):
        if not self.img:
            return

        #self.analisar_composicao()
        #self.analisar_score()
        self.analisar_quebra()

    def analisar_composicao(self):

        r = ProcessImageFunctions.calcular_composicao(self.img, self.paletas.refs)

        self.lbl_g.config(text=f"Germen:   {r.get('germen', 0):.1f}%")
        self.lbl_c.config(text=f"Casca:    {r.get('casca', 0):.1f}%")
        self.lbl_k.config(text=f"Canjica:  {r.get('canjica', 0):.1f}%")
        self.lbl_u.config(text=f"Util:     {r.get('util', 0):.1f}%")
        self.lbl_f.config(text=f"Exclusao: {r.get('exclusao', 0):.1f}%")
    

    def analisar_score(self):
        if not self.img:
            return
        score_info = ProcessImageFunctions.calcular_score(self.img, self.paletas.refs)
        score_val = score_info.get("score", 0)
        self.lbl_deger.config(text=f"Score {score_val:.2f}")

    def analisar_quebra(self):
        if not self.img:
            return
        q = ProcessImageFunctions.avaliar_quebra(self.img)
        score_q = q.get("score_quebra", 0.0)
        classe = q.get("classe", "indefinido")
        self.lbl_quebra.config(text=f"Quebra: {score_q:.2f} ({classe})")

    
