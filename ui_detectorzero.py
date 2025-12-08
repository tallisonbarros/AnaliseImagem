import tkinter as tk
import tkinter.messagebox as mb
import os
import cv2  # ✔ IMPORT NECESSÁRIO PARA PROCESSAMENTO DAS IMAGENS
import UiFunctions
import FileFunctions
import ImageFunctions
from PIL import Image, ImageTk

imagem_atual_caminho = None
label_imagem_global = None
imagem_tk_global = None
botoes_classificacao = []

def limpar_ui_fim():
    global label_imagem_global

    # Limpa imagem
    label_imagem_global.config(image="", text="")

    # Remove completamente os botões
    for widget in botoes_classificacao:
        widget.grid_remove()


def carregar_proxima_nao_classificada():
    global imagem_atual_caminho, label_imagem_global

    try:
        # 1) Criar objeto Pasta para ImgNotClass
        pasta_nc = FileFunctions.Pasta("ImgNotClass")

        # 2) Filtrar apenas PNG (por enquanto)
        pasta_nc.filtrar_arquivos(".png")

        # 3) Verificar se há arquivos
        if not pasta_nc.lista_arquivos:
            mb.showinfo("Fim", "Nenhuma imagem não classificada encontrada.")
            imagem_atual_caminho = None
            limpar_ui_fim()
            return

        # 4) Pegar o primeiro arquivo
        caminho = pasta_nc.lista_arquivos[0]
        imagem_atual_caminho = caminho

        # 5) Criar o objeto Imagem
        img = ImageFunctions.Imagem(caminho)

        if not img.valida:
            mb.showerror("Erro", img.erro)
            return

        # 6) Exibir imagem na UI
        UiFunctions.exibir_imagem_ui(img.matriz_NumPy, label_imagem_global)

        # 7) Mostrar novamente os botões de classificação
        for widget in botoes_classificacao:
            widget.grid()

    except Exception as e:
        mb.showerror("Erro inesperado", str(e))




def classificar_imagem(classe):
    global imagem_atual_caminho

    if not imagem_atual_caminho:
        mb.showwarning("Aviso", "Nenhuma imagem carregada para classificar.")
        return

    try:
        destino_pasta = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ImgDataBase",
            str(classe)
        )

        resultado = FileFunctions.MoverArquivo(imagem_atual_caminho, destino_pasta)

        if not resultado["ok"]:
            mb.showerror("Erro ao mover arquivo", resultado["erro"])
            return

        # Feedback ao usuário
        mb.showinfo(
            "Classificado!",
            f"A imagem foi movida para a classe {classe}.\n\nDestino:\n{resultado['destino']}"
        )

        imagem_atual_caminho = None

        carregar_proxima_nao_classificada()

    except Exception as e:
        mb.showerror("Erro inesperado", str(e))

def criar_janela():
    global label_imagem_global

    root = tk.Tk()
    root.title("DetectorZero - UI Simples")

    root.minsize(300, 150)

    label = tk.Label(
        root,
        text="DetectorZero - Protótipo de Interface",
        font=("Arial", 12)
    )
    label.pack(pady=10)

    campo_texto = tk.Entry(root, width=40)
    campo_texto.pack(pady=5)

    botao_ler_pasta = tk.Button(
        root,
        text="Ver conteudo da pasta",
        command=lambda: UiFunctions.ler_pasta(campo_texto.get())
    )
    botao_ler_pasta.pack(pady=10)

    botao_proxima_nc = tk.Button(
        root,
        text="Próxima imagem não classificada",
        command=carregar_proxima_nao_classificada
    )
    botao_proxima_nc.pack(pady=5)

    # ✔ ÁREA QUE VAI EXIBIR IMAGEM
    label_imagem_global = tk.Label(root)
    label_imagem_global.pack(pady=10)

    frame_classes = tk.Frame(root)
    frame_classes.pack(pady=10)

    for i in range(1, 11):
        btn = tk.Button(
            frame_classes,
            text=str(i),
            width=4,
            command=lambda n=i: classificar_imagem(n)
        )
        btn.grid(row=0, column=i)
        botoes_classificacao.append(btn)  # ← AGORA DENTRO DO for

    info = tk.Label(
        root,
        text="Clique em 'Próxima imagem' para começar a classificar.",
        font=("Arial", 9)
    )
    info.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    criar_janela()
