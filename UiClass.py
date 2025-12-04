# ui_classificador.py
import tkinter as tk
import FileFunctions
import ImageFunctions
import UiFunctions

class ClassificadorUI:
    def __init__(self, root, dirDataBase):
        self.root = root
        self.dirDataBase = dirDataBase
        self.imagem_atual = None

        # Label onde exibe imagem
        self.label_imagem = tk.Label(root)
        self.label_imagem.pack(pady=10)

        # Botão buscar
        botao_buscar = tk.Button(
            root,
            text="Buscar imagem",
            command=self.buscar_imagem_not_class   # 👈 sem parâmetros
        )
        botao_buscar.pack(pady=5)

    def buscar_imagem_not_class(self):
        pasta_nc = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
        pasta_nc.filtrar_arquivos(".png")

        if not pasta_nc.lista_arquivos:
            # aqui pode usar messagebox, ou delegar para UiFunctions
            from tkinter import messagebox
            messagebox.showinfo("Fim", "Nenhuma imagem não classificada encontrada.")
            self.imagem_atual = None
            return

        caminho = pasta_nc.lista_arquivos[0]
        img = ImageFunctions.Imagem(caminho)

        if not img.valida:
            from tkinter import messagebox
            messagebox.showerror("Erro", img.erro)
            return

        UiFunctions.exibir_imagem_ui(img.matriz_NumPy, self.label_imagem)
        self.imagem_atual = caminho   # estado da tela

