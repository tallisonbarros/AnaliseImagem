# ui_classificador.py
import tkinter as tk
import tkinter.messagebox as mb
import FileFunctions
import ImageFunctions
import ProcessImageFunctions
import UiFunctions

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
            command=self.buscar_imagem_not_class   # 👈 sem parâmetros
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
        

        self.root.after(300, self.buscar_imagem_not_class)
        
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
 
    def buscar_imagem_not_class(self):
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

            analise = ProcessImageFunctions.analisar(img.matriz_NumPy)
            self.atualizar_painel_analise(analise)




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

            self.buscar_imagem_not_class()
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
        
        self.buscar_imagem_not_class()


    def atualizar_painel_analise(self, resultado):
        casca   = resultado.get("casca", 0.0)
        canjica = resultado.get("canjica", 0.0)
        germen  = resultado.get("germen", 0.0)

        self.lbl_casca.config(  text=f"Casca:   {casca:.1f} %")
        self.lbl_canjica.config(text=f"Canjica: {canjica:.1f} %")
        self.lbl_germen.config( text=f"Gérmen:  {germen:.1f} %")


    
    def analisar (self):
            resultado = ProcessImageFunctions.analisar(self)
            self.atualizar_painel_analise(resultado)

