# ------------------- UiCalibrarQuebra.py
import os
import tkinter as tk
import tkinter.messagebox as mb


class CalibrarQuebra:
    """
    Placeholder de tela para calibracao de quebra.
    Permite treinar/gerar o modelo cnn_face_interna.pt a partir de um dataset anotado.
    """
    def __init__(self, root, base_dir=None):
        self.root = root
        self.base_dir = base_dir
        self.dataset_dir = tk.StringVar()
        self.output_path = tk.StringVar(
            value=os.path.join(self.base_dir or os.getcwd(), "cnn_quebra.pt")
        )
        self.dataset_out = tk.StringVar(
            value=os.path.join(self.base_dir or os.getcwd(), "dataset_quebra")
        )
        self.epochs = tk.IntVar(value=5)
        self.batch_size = tk.IntVar(value=8)
        self.lr = tk.DoubleVar(value=1e-3)
        self.tamanho = tk.IntVar(value=224)
        self._status = tk.StringVar(value="Dataset rotulado (0/1/2) -> treinar modelo.")
        self._btn_treinar = None
        self._montar_ui()

    def _montar_ui(self):
        container = tk.Frame(self.root, padx=20, pady=20)
        container.pack(fill="both", expand=True)
        tk.Label(container, text="Calibrar Quebra", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        # dataset
        linha_ds = tk.Frame(container)
        linha_ds.pack(fill="x", pady=4)
        tk.Label(linha_ds, text="Pasta dataset (0/1/2 ou integro/parcial/quebrado):").pack(side="left")
        tk.Entry(linha_ds, textvariable=self.dataset_dir, width=40).pack(side="left", padx=6)
        tk.Button(linha_ds, text="...", command=self._selecionar_pasta).pack(side="left")

        # saida
        linha_out = tk.Frame(container)
        linha_out.pack(fill="x", pady=4)
        tk.Label(linha_out, text="Saida modelo (.pt):").pack(side="left")
        tk.Entry(linha_out, textvariable=self.output_path, width=40).pack(side="left", padx=6)

        linha_ds_out = tk.Frame(container)
        linha_ds_out.pack(fill="x", pady=4)
        tk.Label(linha_ds_out, text="Pasta destino dataset:").pack(side="left")
        tk.Entry(linha_ds_out, textvariable=self.dataset_out, width=40).pack(side="left", padx=6)
        tk.Button(linha_ds_out, text="Criar dataset", command=self._abrir_construtor_dataset).pack(side="left", padx=4)
        tk.Label(
            container,
            text="Anotador: 0=integro | 1=parcial | 2=quebrado | n=pular | q/Esc=sair",
            fg="#333"
        ).pack(pady=(2, 8))

        # parametros
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

    def _selecionar_pasta(self):
        from tkinter import filedialog as fd
        pasta = fd.askdirectory(title="Selecione a pasta do dataset")
        if pasta:
            self.dataset_dir.set(pasta)

    def _treinar(self):
        try:
            import cnn_trainer
        except Exception as exc:  # noqa: BLE001
            mb.showerror("Treino", f"Modulo cnn_trainer nao disponivel: {exc}")
            return
        if not getattr(cnn_trainer, "HAS_TORCH", False):
            mb.showerror("Treino", "PyTorch nao instalado. Instale para treinar o modelo.")
            return

        ds = self.dataset_dir.get().strip()
        saida = self.output_path.get().strip()
        if not ds:
            mb.showerror("Treino", "Informe a pasta do dataset.")
            return
        if not saida:
            mb.showerror("Treino", "Informe o caminho de saida do modelo (.pt).")
            return

        self._status.set("Treinando... aguarde.")
        self._btn_treinar.config(state="disabled")
        self.root.update_idletasks()

        import threading

        def worker():
            try:
                res = cnn_trainer.treinar_modelo(
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

    def _abrir_construtor_dataset(self):
        try:
            import cnn_trainer
        except Exception as exc:  # noqa: BLE001
            mb.showerror("Dataset", f"Modulo cnn_trainer nao disponivel: {exc}")
            return
        pasta_origem = self.dataset_dir.get().strip()
        pasta_dest = self.dataset_out.get().strip()
        if not pasta_origem:
            mb.showerror("Dataset", "Informe a pasta de origem das imagens.")
            return
        if not pasta_dest:
            mb.showerror("Dataset", "Informe a pasta de destino do dataset.")
            return
        self._status.set("Abrindo construtor de dataset...")
        self.root.update_idletasks()

        import threading

        def worker():
            try:
                cnn_trainer.construir_dataset(pasta_origem, pasta_dest, tamanho=max(32, self.tamanho.get()))
                msg = f"Dataset gerado em {pasta_dest}"
                res_ok = True
            except Exception as exc:  # noqa: BLE001
                msg = f"Falha: {exc}"
                res_ok = False

            def finalizar():
                if res_ok:
                    mb.showinfo("Dataset", msg)
                    self.dataset_dir.set(pasta_dest)
                else:
                    mb.showerror("Dataset", msg)
                self._status.set(msg)

            self.root.after(0, finalizar)

        threading.Thread(target=worker, daemon=True).start()
