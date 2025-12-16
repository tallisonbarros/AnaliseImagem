# ui_classificador.py
import tkinter as tk
import UiClassificador
import UiCalibrarQuebra
import os
import tkinter.messagebox as mb


if __name__ == "__main__":
    import sys
    print("PYTHON EXECUTADO:", sys.executable)
    print("PYTHONPATH:", sys.path)
  #  mb.showinfo("Título", "Iniciado")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    root = tk.Tk()  # Cria Tela
    root.title("Yris Studio - Menu")  # Nomeia Tela
    menu_frame = tk.Frame(root, padx=20, pady=20)
    menu_frame.pack()

    def abrir_classificador():
        menu_frame.destroy()
        root.title("Yris Studio - Classificador")
        UiClassificador.ClassificadorUI(root, base_dir)  # Configura tela principal

    def abrir_calibrar_quebra():
        menu_frame.destroy()
        root.title("Yris Studio - Calibrar Quebra")
        UiCalibrarQuebra.CalibrarQuebra(root, base_dir)

    tk.Label(menu_frame, text="Selecione o módulo", font=("Arial", 12, "bold")).pack(pady=(0, 10))
    tk.Button(menu_frame, text="Classificador", width=20, command=abrir_classificador).pack(pady=5)
    tk.Button(menu_frame, text="Calibrar Quebra", width=20, command=abrir_calibrar_quebra).pack(pady=5)

    root.mainloop()  # Roda Tela
