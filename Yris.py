# ui_classificador.py
import tkinter as tk
import UiClassificador
import UiCalibrarQuebra
import UiCalibrarVisao
import os
import tkinter.messagebox as mb


if __name__ == "__main__":
    import sys
    print("PYTHON EXECUTADO:", sys.executable)
    print("PYTHONPATH:", sys.path)
  #  mb.showinfo("TA-tulo", "Iniciado")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    root = tk.Tk()  # Cria Tela

    def mostrar_menu():
        for w in root.winfo_children():
            w.destroy()
        root.title("Yris Studio - Menu")  # Nomeia Tela
        menu_frame = tk.Frame(root, padx=20, pady=20)
        menu_frame.pack()

        def abrir_classificador():
            for w in root.winfo_children():
                w.destroy()
            root.title("Yris Studio - Classificador")
            UiClassificador.ClassificadorUI(root, base_dir, on_back=mostrar_menu)  # Configura tela principal

        def abrir_calibrar_quebra():
            for w in root.winfo_children():
                w.destroy()
            root.title("Yris Studio - Calibrar Quebra")
            UiCalibrarQuebra.CalibrarQuebra(root, base_dir, on_back=mostrar_menu)

        def abrir_calibrar_visao():
            for w in root.winfo_children():
                w.destroy()
            root.title("Yris Studio - Calibrar Visao")
            UiCalibrarVisao.CalibrarVisao(root, base_dir, on_back=mostrar_menu)

        tk.Label(menu_frame, text="Selecione o modulo", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        tk.Button(menu_frame, text="Classificador", width=20, command=abrir_classificador).pack(pady=5)
        tk.Button(menu_frame, text="Calibrar Quebra", width=20, command=abrir_calibrar_quebra).pack(pady=5)
        tk.Button(menu_frame, text="Calibrar Visao", width=20, command=abrir_calibrar_visao).pack(pady=5)

    mostrar_menu()
    root.mainloop()  # Roda Tela
