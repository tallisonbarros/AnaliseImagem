# ui_classificador.py
import tkinter as tk
import UiClass


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirDataBase = os.path.join(base_dir, "ImgDataBase")

    root = tk.Tk()
    root.title("Classificador de Imagens")

    app = UiClass.ClassificadorUI(root, dirDataBase)
    root.mainloop()
