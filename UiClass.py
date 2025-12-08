# ui_classificador.py
import tkinter as tk
import tkinter.messagebox as mb
import FileFunctions
import ImageFunctions
import ProcessImageFunctions
import UiFunctions
import numpy as np
import cv2


class ClassificadorUI:
    def __init__(self, root, dirDataBase):
        self.root = root
        self.dirDataBase = dirDataBase
        self.imagem_atual = None
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        self.root.eval('tk::PlaceWindow . center')


        
        # Botão buscar
        botao_buscar = tk.Button(
            self.root,
            text="Buscar imagem",
            command=self.buscar_imagem_notclass   # 👈 sem parâmetros
        )
        # botao_buscar.pack(pady=5)
        
        # Botão Desclassificar tudo
        botao_desclas = tk.Button(
            self.root,
            text="Desclassificar Tudo",
            command=self.desclassificar_tudo   # 👈 sem parâmetros
        )
        botao_desclas.pack(pady=5)

        # Label onde exibe imagem
        self.label_imagem = tk.Label(self.root)
        self.label_imagem.bind("<Button-1>", self.click_imagem)


        # Texto onde exibe directorio da imagem atual
        self.TxtDirImagem = tk.Label(
            self.root,
            font=("Arial", 12)
        )

        # Quantidade de itens para classificar:
        self.TxtQtdItens = tk.Label(
            self.root,
            font=("Arial", 8)
        )

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

        

        self.root.after(300, self.buscar_imagem_notclass)
        
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

        # Frame da Pre Processamento
        frame_preprocess = tk.Frame(self.root)
        frame_preprocess.pack(side="right", padx=10, pady=10)

        titulo = tk.Label(frame_preprocess, text="Pre Processamento", font=("Arial", 10, "bold"))
        titulo.pack()
        self.preprocess_inputs = UiFunctions.criar_input_preprocess(frame_preprocess)
        
        # self.HSVcor = tk.Label(frame_preprocess, text="Cor:", font=("Arial", 6, "bold"))
       # self.HSVcor.pack()

 
    def buscar_imagem_notclass(self):
            self.pasta_nc = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
            self.pasta_nc.filtrar_arquivos(".png")

            if not self.pasta_nc.lista_arquivos:
                # aqui pode usar messagebox, ou delegar para UiFunctions
                mb.showinfo("Fim", "Nenhuma imagem não classificada encontrada.")
                self.imagem_atual = None
                self.label_imagem.pack_forget()
                self.frame_classes.pack_forget()
                self.TxtDirImagem.pack_forget()
                self.botao_analisar.pack_forget()
                return
            caminho = self.pasta_nc.lista_arquivos[0]
            img = ImageFunctions.Imagem(caminho)

            if not img.valida:
                mb.showerror("Erro", img.erro)
                return
            
            self.label_imagem.pack(pady=10)
            self.TxtDirImagem.pack(pady=10)
            self.frame_classes.pack(pady=10)
            self.TxtQtdItens.pack(pady=10)
            self.botao_analisar.pack(pady=10)

            UiFunctions.exibir_imagem_ui(img.matriz_NumPy, self.label_imagem)
            self.imagem_atual = caminho   # estado da tela
            self.TxtDirImagem.config(text=img.nome)
            self.TxtQtdItens.config(text=str(self.pasta_nc.quantidade_arquivos) + " imagens para classificar.")

            self.analisar()





    def classificar_imagem(self, classe):
        if not self.imagem_atual :
            mb.showinfo("Classificar_imagem", "Imagem nao encontrada")
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

    def desclassificar_tudo(self):
        # Pega o caminho de ImgNotClass
        pasta_nc = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
        dir_nc = pasta_nc.dir

        total_movidos = 0

        # Loop nas classes 1..10
        for classe in range(1, 11):
            pasta_classe = FileFunctions.Pasta(self.dirDataBase, str(classe))

            if not pasta_classe.lista_arquivos:
                continue

            for caminho in pasta_classe.lista_arquivos:
                resultado = FileFunctions.MoverArquivo(caminho, dir_nc)
                if resultado["ok"]:
                    total_movidos += 1

        # Limpa estado da tela
        self.imagem_atual = None
        self.label_imagem.config(image="", text="")

        # Feedback
        if total_movidos == 0:
            mb.showinfo("Desclassificar tudo", "Nenhuma imagem para desclassificar.")
        else:
            mb.showinfo(
                "Desclassificar tudo",
                f"{total_movidos} imagens foram movidas para ImgNotClass."
            )
            # Recarrega fluxo normal
        
        self.buscar_imagem_notclass()


    def atualizar_painel_analise(self, resultado):
        casca   = resultado.get("casca", 0.0)
        canjica = resultado.get("canjica", 0.0)
        germen  = resultado.get("germen", 0.0)

        self.lbl_casca.config(  text=f"Casca:   {casca:.1f} %")
        self.lbl_canjica.config(text=f"Canjica: {canjica:.1f} %")
        self.lbl_germen.config( text=f"Gérmen:  {germen:.1f} %")


    
    def analisar (self):            
            img = ImageFunctions.Imagem(self.imagem_atual)
            #print("DEBUG inputs", self.preprocess_inputs)
            resultado = ProcessImageFunctions.analisar(img, UiFunctions.coletar_hsv(self.preprocess_inputs))
            print("DEBUG RESULTADO:", resultado)

            self.atualizar_painel_analise(resultado)




    def click_imagem(self, event):
        if self.imagem_atual is None:
            return

        # 1. Carrega imagem novamente
        img = ImageFunctions.Imagem(self.imagem_atual)
        matriz = img.matriz_NumPy

        # Dimensões reais da imagem
        h_real, w_real = matriz.shape[:2]

        # Dimensões exibidas (350x350)
        w_exib, h_exib = 350, 350

        # 2. Converte coordenadas da UI → coordenadas reais
        x = int(event.x * (w_real / w_exib))
        y = int(event.y * (h_real / h_exib))

        if x >= w_real or y >= h_real:
            return

        # 3. Captura pixel BGR
        b, g, r = matriz[y, x]

        # 4. Converte para HSV
        pixel_hsv = cv2.cvtColor(
            np.uint8([[[b, g, r]]]),
            cv2.COLOR_BGR2HSV
        )[0][0]

        H, S, V = pixel_hsv

        # 5. Atualiza o texto na interface
        self.preprocess_inputs["HSVCor"].config(text=f"Cor HSV: H={H}  S={S}  V={V}")
