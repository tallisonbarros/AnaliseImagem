# ------------------- UiCalibrarVisao.py
import os
import tkinter as tk
import tkinter.messagebox as mb
import FileFunctions


class CalibrarVisao:
    """
    Tela para treinar o classificador de visao (notas 1..10).
    Dataset assume pastas ja rotuladas (1..10).
    """
    def __init__(self, root, base_dir=None, on_back=None):
        self.root = root
        self.base_dir = base_dir
        self.on_back = on_back
        self.root_dir = tk.StringVar(value=self.base_dir or os.getcwd())
        self.dataset_dir = tk.StringVar(value=os.path.join(self.root_dir.get(), "dataset_visao"))
        self.output_path = tk.StringVar(value=os.path.join(self.root_dir.get(), "cnn_visao.pt"))
        self.epochs = tk.IntVar(value=5)
        self.batch_size = tk.IntVar(value=8)
        self.lr = tk.DoubleVar(value=1e-3)
        self.tamanho = tk.IntVar(value=224)
        self._status = tk.StringVar(value="Dataset 1..10 -> treinar modelo.")
        self._btn_treinar = None
        self._montar_ui()

    def _montar_ui(self):
        container = tk.Frame(self.root, padx=20, pady=20)
        container.pack(fill="both", expand=True)
        tk.Label(container, text="Treinar Visao (1..10)", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        linha_top = tk.Frame(container)
        linha_top.pack(fill="x", pady=(0, 6))
        tk.Button(linha_top, text="Voltar ao menu", command=self._voltar_menu).pack(side="left")

        linha_raiz = tk.Frame(container)
        linha_raiz.pack(fill="x", pady=4)
        tk.Label(linha_raiz, text="Pasta raiz:").pack(side="left")
        tk.Entry(linha_raiz, textvariable=self.root_dir, width=40, state="readonly").pack(side="left", padx=6)
        tk.Button(linha_raiz, text="Pasta raiz", command=self._selecionar_pasta_raiz).pack(side="left")

        linha_ds = tk.Frame(container)
        linha_ds.pack(fill="x", pady=4)
        tk.Label(linha_ds, text="Pasta dataset (1..10):").pack(side="left")
        tk.Entry(linha_ds, textvariable=self.dataset_dir, width=40, state="readonly").pack(side="left", padx=6)
        tk.Button(linha_ds, text="Alterar", command=self._selecionar_dataset).pack(side="left", padx=4)

        linha_out = tk.Frame(container)
        linha_out.pack(fill="x", pady=4)
        tk.Label(linha_out, text="Saida modelo (.pt):").pack(side="left")
        tk.Entry(linha_out, textvariable=self.output_path, width=40, state="readonly").pack(side="left", padx=6)
        tk.Button(linha_out, text="Alterar", command=self._selecionar_saida).pack(side="left", padx=4)

        linha_params = tk.Frame(container)
        linha_params.pack(fill="x", pady=6)
        tk.Label(linha_params, text="Epochs:").pack(side="left")
        tk.Entry(linha_params, textvariable=self.epochs, width=5).pack(side="left", padx=4)
        tk.Label(linha_params, text="Batch:").pack(side="left")
        tk.Entry(linha_params, textvariable=self.batch_size, width=5).pack(side="left", padx=4)
        tk.Label(linha_params, text="LR:").pack(side="left")
        tk.Entry(linha_params, textvariable=self.lr, width=8).pack(side="left", padx=4)
        tk.Label(linha_params, text="Tam:").pack(side="left")
        tk.Entry(linha_params, textvariable=self.tamanho, width=5).pack(side="left", padx=4)

        self._btn_treinar = tk.Button(container, text="Treinar modelo", command=self._treinar)
        self._btn_treinar.pack(pady=10)

        tk.Label(container, textvariable=self._status, fg="blue").pack(pady=(6, 0))

    def _selecionar_pasta_raiz(self):
        from tkinter import filedialog as fd
        pasta = fd.askdirectory(title="Selecione a pasta raiz")
        if pasta:
            self.root_dir.set(pasta)
            self.dataset_dir.set(os.path.join(pasta, "dataset_visao"))
            self.output_path.set(os.path.join(pasta, "cnn_visao.pt"))
            self._status.set(f"Pasta raiz definida: {pasta}")

    def _selecionar_dataset(self):
        from tkinter import filedialog as fd
        pasta = fd.askdirectory(title="Selecione a pasta do dataset (1..10)")
        if pasta:
            self.dataset_dir.set(pasta)
            self._status.set(f"Dataset: {pasta}")

    def _selecionar_saida(self):
        from tkinter import filedialog as fd
        caminho = fd.asksaveasfilename(
            title="Salvar modelo (.pt)",
            defaultextension=".pt",
            filetypes=[("TorchScript", "*.pt"), ("Todos", "*.*")]
        )
        if caminho:
            self.output_path.set(caminho)

    def _treinar(self):
        try:
            import cnn_trainer_visao
        except Exception as exc:  # noqa: BLE001
            mb.showerror("Treino", f"Modulo cnn_trainer_visao nao disponivel: {exc}")
            return
        if not getattr(cnn_trainer_visao, "HAS_TORCH", False):
            mb.showerror("Treino", "PyTorch nao instalado. Instale para treinar o modelo.")
            return

        ds = self.dataset_dir.get().strip()
        saida = self.output_path.get().strip()
        if not ds:
            mb.showerror("Treino", "Pasta do dataset nao definida.")
            return
        if not saida:
            mb.showerror("Treino", "Caminho de saida do modelo (.pt) nao definido.")
            return

        self._status.set("Treinando... aguarde.")
        self._btn_treinar.config(state="disabled")
        self.root.update_idletasks()

        import threading

        def worker():
            try:
                res = cnn_trainer_visao.treinar_modelo(
                    ds,
                    saida,
                    epochs=max(1, self.epochs.get()),
                    batch_size=max(1, self.batch_size.get()),
                    lr=float(self.lr.get()),
                    tamanho=max(32, self.tamanho.get()),
                )
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "erro": str(exc)}

            def finalizar():
                if res.get("ok"):
                    mb.showinfo("Treino", f"Modelo salvo em: {res.get('modelo')}")
                    self._status.set(f"OK: {res.get('modelo')} (ep={res.get('epochs')}, samples={res.get('amostras')})")
                else:
                    mb.showerror("Treino", f"Falha: {res.get('erro')}")
                    self._status.set(f"Erro: {res.get('erro')}")
                self._btn_treinar.config(state="normal")

            self.root.after(0, finalizar)

        threading.Thread(target=worker, daemon=True).start()

    def _voltar_menu(self):
        if self.on_back:
            self.on_back()
        else:
            self.root.destroy()
