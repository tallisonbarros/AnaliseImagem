import tkinter as tk
import tkinter.messagebox as mb
import FileFunctions
import ImageFunctions
import ProcessImageFunctions
import UiFunctions
import os
import numpy as np
import cv2
import PIL
from PIL import Image, ImageTk

class ClassificadorUI:
    def __init__(self, root, base_dir):
        self.root = root
        self.dirDataBase = os.path.join(base_dir, "ImgDataBase")
        self.imagem_atual = None
        self.img_bgr_atual = None
        caminho_logo = os.path.join(base_dir, "imagens/logos/SET_FAVICON.png")

        self.root.geometry("1000x800")
        self.root.resizable(False, False)
        self.root.eval('tk::PlaceWindow . center')

        self.img_original = None  

        header = tk.Frame(self.root)
        header.pack(fill="x", pady=0, anchor="n")

        logo = Image.open(caminho_logo)
        logo = logo.resize((120, 120))  
        self.logo_tk = ImageTk.PhotoImage(logo)

        label_logo = tk.Label(header, image=self.logo_tk)
        label_logo.pack(pady=0)


        # Canvas onde exibe imagem
        self.canvas = tk.Canvas(self.root, width=300, height=255, bg="black")
        self.canvas.place(x=20, y=40)

        # Navegador de Zoom/Pan
        self.navigator = UiFunctions.CanvasNavigator(self.canvas)

        # Pegar cor (com add="+" para coexistir com pan)
        self.canvas.bind("<Button-3>", self.pegar_cor, add="+")

        # Texto onde exibe o diretório da imagem atual
        self.TxtDirImagem = tk.Label(self.root, font=("Arial", 12))
        self.TxtQtdItens   = tk.Label(self.root, font=("Arial", 8))

        # Frame da análise automática
        frame_analise = tk.Frame(self.root)
        frame_analise.pack(side="right", padx=10, pady=10)

        titulo2 = tk.Label(frame_analise, text="Análise automática", font=("Arial", 10, "bold"))
        titulo2.pack()

        self.lbl_germen  = tk.Label(frame_analise, text="Gérmen: -- %")
        self.lbl_casca   = tk.Label(frame_analise, text="Casca: -- %")
        self.lbl_canjica = tk.Label(frame_analise, text="Canjica: -- %")

        self.lbl_germen.pack(anchor="w")
        self.lbl_casca.pack(anchor="w")
        self.lbl_canjica.pack(anchor="w")

        # Frame classes
        self.frame_classes = tk.Frame(self.root)

        for i in range(1, 11):
            btn = tk.Button(
                self.frame_classes,
                text=str(i),
                width=4,
                command=lambda n=i: self.classificar_imagem(n)
            )
            btn.grid(row=0, column=i)

        # Botão Analisar
        self.botao_analisar = tk.Button(
            self.root,
            text="Analisar",
            command=self.analisar
        )

        # Frame de Pré-Processamento
        frame_preprocess = tk.Frame(self.root)
        frame_preprocess.pack(side="right", padx=10, pady=10)

        titulo = tk.Label(frame_preprocess, text="Pré-Processamento", font=("Arial", 10, "bold"))
        titulo.pack()

        # Botão Desclassificar Tudo
        self.botao_desclas = tk.Button(
            self.root,
            text="Desclassificar Tudo",
            command=self.desclassificar_tudo
        )
        
        # Paleta de cores
        self.paletas = UiFunctions.PaletaCategoriaCores(self.get_pickcolor, self.on_paleta_alterada)
        self.paletas.criar_paletas(frame_preprocess)  
        
        self.root.after(300, self.buscar_imagem_notclass)

        # Paleta de cores nao categorizadas
        self.frame_cores_livres = tk.LabelFrame(
            frame_preprocess,
            text="Cores não categorizadas",
            font=("Arial", 9, "bold")
        )
        self.frame_cores_livres.pack(pady=5, fill="x")


    # ============================================================
    # BUSCAR PRÓXIMA IMAGEM NÃO CLASSIFICADA
    # ============================================================
    def buscar_imagem_notclass(self):
        self.pasta_nc = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
        self.pasta_nc.filtrar_arquivos(".png")

        if not self.pasta_nc.lista_arquivos:
            mb.showinfo("Fim", "Nenhuma imagem não classificada encontrada.")
            self.imagem_atual = None
            self.canvas.pack_forget()
            self.frame_classes.pack_forget()
            self.TxtDirImagem.pack_forget()
            self.botao_analisar.pack_forget()
            return

        caminho = self.pasta_nc.lista_arquivos[0]
        img = ImageFunctions.Imagem(caminho)
        self.img_bgr_atual = img.matriz_NumPy.copy()


        if not img.valida:
            mb.showerror("Erro", img.erro)
            return

        self.canvas.pack(pady=10)
        self.TxtDirImagem.pack(pady=10)
        self.frame_classes.pack(pady=10)
        self.TxtQtdItens.pack(pady=10)
        self.botao_analisar.pack(pady=10)

        # Converte para PIL
        rgb = cv2.cvtColor(img.matriz_NumPy, cv2.COLOR_BGR2RGB)
        self.img_original = PIL.Image.fromarray(rgb)

        # Envia para o navigator (ele redesenha e reseta zoom/pan)
        self.navigator.set_image(self.img_original)

        self.imagem_atual = caminho
        self.TxtDirImagem.config(text=img.nome)
        self.TxtQtdItens.config(text=str(self.pasta_nc.quantidade_arquivos) + " imagens para classificar.")
        
        self.botao_desclas.pack(pady=5)
        
        self.atualizar_cores_nao_categorizadas(img.matriz_NumPy)
        self.analisar()


    # ============================================================
    # CLASSIFICAR IMAGEM
    # ============================================================
    def classificar_imagem(self, classe):
        if not self.imagem_atual:
            mb.showinfo("Classificar_imagem", "Imagem não encontrada")
            return

        try:
            pasta_classe = FileFunctions.Pasta(self.dirDataBase, str(classe))
            destino_pasta = pasta_classe.dir

            resultado = FileFunctions.MoverArquivo(self.imagem_atual, destino_pasta)
            if not resultado["ok"]:
                return

            self.buscar_imagem_notclass()
            return "OK"

        except Exception as e:
            return


    # ============================================================
    # DESCLASSIFICAR TODAS AS IMAGENS
    # ============================================================
    def desclassificar_tudo(self):
        pasta_nc = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
        dir_nc = pasta_nc.dir

        total_movidos = 0

        for classe in range(1, 11):
            pasta_classe = FileFunctions.Pasta(self.dirDataBase, str(classe))

            if not pasta_classe.lista_arquivos:
                continue

            for caminho in pasta_classe.lista_arquivos:
                resultado = FileFunctions.MoverArquivo(caminho, dir_nc)
                if resultado["ok"]:
                    total_movidos += 1

        # limpa a imagem da tela
        self.imagem_atual = None

        self.canvas.delete("all")              # 👍 correto
        self.navigator.img_original = None     # reseta imagem
        self.navigator.img_tk = None
        self.navigator.zoom = 1.0
        self.navigator.pan_x = 0
        self.navigator.pan_y = 0

        if total_movidos == 0:
            mb.showinfo("Desclassificar tudo", "Nenhuma imagem para desclassificar.")
        else:
            mb.showinfo(
                "Desclassificar tudo",
                f"{total_movidos} imagens movidas para ImgNotClass."
            )

        self.buscar_imagem_notclass()



    # ============================================================
    # PEGAR COR NO CLIQUE
    # ============================================================
    def pegar_cor(self, event):
        if self.imagem_atual is None:
            return

        # converte coordenada canvas → imagem original
        coords = self.navigator.canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return

        x, y = coords

        img = ImageFunctions.Imagem(self.imagem_atual)
        matriz = img.matriz_NumPy
        h_real, w_real = matriz.shape[:2]

        if x < 0 or y < 0 or x >= w_real or y >= h_real:
            return

        b, g, r = matriz[y, x]

        pixel_hsv = cv2.cvtColor(
            np.uint8([[[b, g, r]]]),
            cv2.COLOR_BGR2HSV
        )[0][0]

        H, S, V = pixel_hsv
        
        import colorsys

        # Conversão HSV → RGB (Tkinter usa RGB)
        r, g, b = colorsys.hsv_to_rgb(H/179, S/255, V/255)
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        cor_hex = f"#{r:02x}{g:02x}{b:02x}"

        self.pickcolor = pixel_hsv
        self.paletas.HSVcor.config(bg=cor_hex)

    # ============================================================
    # ANÁLISE AUTOMÁTICA
    # ============================================================
    def atualizar_painel_analise(self, resultado):
        casca   = resultado.get("casca", 0.0)
        canjica = resultado.get("canjica", 0.0)
        germen  = resultado.get("germen", 0.0)

        self.lbl_casca.config(  text=f"Casca:   {casca:.1f} %")
        self.lbl_canjica.config(text=f"Canjica: {canjica:.1f} %")
        self.lbl_germen.config( text=f"Gérmen:  {germen:.1f} %")


    def analisar(self):
        img = ImageFunctions.Imagem(self.imagem_atual)

        resultado = ProcessImageFunctions.processar_valores_hsv(
            img,
            UiFunctions.coletar_hsv(self.paletas.refs)
        )

        self.atualizar_painel_analise(resultado)


    def get_pickcolor(self):
        return getattr(self, "pickcolor", None)

    def atualizar_cores_nao_categorizadas(self, imagem_bgr):
        """
        Limpa o frame_cores_livres e desenha quadradinhos
        com as cores que aparecem na imagem e não pertencem
        a nenhuma categoria cadastrada.
        """
        # limpa o frame
        for w in self.frame_cores_livres.winfo_children():
            w.destroy()

        if imagem_bgr is None:
            return

        # pega lista de cores livres (HSV)
        cores_livres = UiFunctions.extrair_cores_nao_categorizadas(
            imagem_bgr,
            self.paletas.refs
        )

        if not cores_livres:
            tk.Label(self.frame_cores_livres, text="(nenhuma cor livre)").pack()
            return

        linha = tk.Frame(self.frame_cores_livres)
        linha.pack()

        colunas_por_linha = 12
        col = 0

        import colorsys

        for (H, S, V) in cores_livres:
            hsv = (H, S, V)

            # HSV → RGB → hex
            r, g, b = colorsys.hsv_to_rgb(H/179, S/255, V/255)
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            cor_hex = f"#{r:02x}{g:02x}{b:02x}"

            quad = tk.Label(linha, bg=cor_hex, width=2, height=1, relief="ridge")
            quad.grid(row=0, column=col, padx=1, pady=1)

            # >>> aqui vem a mágica: clicar no quadrado carrega a cor
            quad.bind(
                "<Button-1>",
                lambda e, hsv=hsv, cor_hex=cor_hex: self._carregar_cor_livre(hsv, cor_hex)
            )

            col += 1
            if col >= colunas_por_linha:
                linha = tk.Frame(self.frame_cores_livres)
                linha.pack()
                col = 0

    def on_paleta_alterada(self):
        # só atualiza se tiver imagem carregada
        if self.imagem_atual is None:
                return
        # aqui você chama sua função que redesenha o frame
        # de cores não categorizadas
        self.atualizar_cores_nao_categorizadas(self.img_bgr_atual)


    def _carregar_cor_livre(self, hsv, cor_hex):
        # guarda HSV para o botão "+" usar depois
        self.pickcolor = hsv

        # atualiza o quadradinho de cor atual
        self.paletas.HSVcor.config(bg=cor_hex)
