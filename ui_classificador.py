# ui_classificador.py
import tkinter as tk
import UiClass    
import os
import tkinter.messagebox as mb


if __name__ == "__main__":
  #  mb.showinfo("Título", "Iniciado")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    root = tk.Tk() # Cria Tela
    root.title("Classificador de Imagens") # Nomeia Tela

    app = UiClass.ClassificadorUI(root, base_dir) # Configura tela
    root.mainloop() # Roda Tela

