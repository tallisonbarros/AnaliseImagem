# ------------------- UiClass.py
import os
import tkinter as tk
import tkinter.messagebox as mb
from tkinter import filedialog as fd

import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime

import FileFunctions
import ImageFunctions
import ProcessImageFunctions
import UiFunctions
import threading


BG_COLOR = "#5c5c5c"


class ClassificadorUI:
    def __init__(self, root, base_dir, on_back=None):
        self.root = root
        self.dirDataBase = os.path.join(base_dir, "ImgDataBase")
        self.base_dir = base_dir
        self.on_back = on_back

        self.img = None
        self.pickcolor = None
        self.logo_img = None

        self.lbl_g = None
        self.lbl_c = None
        self.lbl_k = None
        self.lbl_u = None
        self.lbl_f = None
        self.lbl_deger = None
        self.lbl_quebra = None
        self.lbl_visao = None
        self.canvas = None
        self.navigator = None
        self.TxtDirImagem = None
        self.TxtQtdItens = None

        self.registrar_score_var = tk.BooleanVar(value=True)
        self.auto_analisar_var = tk.BooleanVar(value=True)

        self._config_root()
        self._build_ui()
        self._carregar_logo()
        self.root.after(300, self.buscar_imagem)

    def _config_root(self):
        self.root.geometry("1100x800")
        self.root.configure(bg=BG_COLOR)
        self.root.option_add("*Background", BG_COLOR)
        for c in range(3):
            self.root.columnconfigure(c, weight=1)
        # Centraliza a janela (usa tamanho atual para evitar posicionar a origem no centro)
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max((sw - w) // 2, 0)
        y = max((sh - h) // 2, 0)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        self._build_header()
        self._build_left_column()
        self._build_paleta_column()
        self._build_analysis_column()

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_COLOR)
        header.grid(row=0, column=0, columnspan=3, pady=10, sticky="we")
        tk.Button(header, text="Voltar ao menu", command=self._voltar_menu).pack(side="left", padx=4)
        self.header = header

    def _build_left_column(self):
        col = tk.Frame(self.root, bg=BG_COLOR)
        col.grid(row=1, column=0, sticky="n")

        self.canvas = tk.Canvas(col, width=250, height=250, bg="black")
        self.canvas.pack(pady=5)
        self.navigator = UiFunctions.CanvasNavigator(self.canvas)
        self.canvas.bind("<Button-3>", self.pegar_cor, add="+")  # click direito captura cor

        self.TxtDirImagem = tk.Label(col, font=("Arial", 12), bg=BG_COLOR)
        self.TxtDirImagem.pack()
        self.TxtQtdItens = tk.Label(col, font=("Arial", 9), bg=BG_COLOR)
        self.TxtQtdItens.pack()

        frame_classes = tk.Frame(col, bg=BG_COLOR)
        frame_classes.pack(pady=8)
        for i in range(1, 11):
            tk.Button(
                frame_classes,
                text=str(i),
                width=4,
                command=lambda n=i: self.classificar_imagem(n)
            ).grid(row=0, column=i)

        bloco_opcoes = tk.LabelFrame(col, text="Ao classificar", bg=BG_COLOR, fg="white")
        bloco_opcoes.pack(fill="x", padx=4, pady=6)
        tk.Checkbutton(
            bloco_opcoes,
            text="Registrar score",
            variable=self.registrar_score_var,
            bg=BG_COLOR,
            activebackground=BG_COLOR,
            selectcolor=BG_COLOR,
            anchor="w",
        ).pack(fill="x", padx=6, pady=1)
        tk.Checkbutton(
            bloco_opcoes,
            text="Auto analisar ao carregar",
            variable=self.auto_analisar_var,
            bg=BG_COLOR,
            activebackground=BG_COLOR,
            selectcolor=BG_COLOR,
            anchor="w",
        ).pack(fill="x", padx=6, pady=1)

        tk.Button(col, text="Analisar Imagem", command=self.analisar_imagem).pack(pady=2)
        tk.Button(col, text="Analisar Composicao", command=self.analisar_composicao).pack(pady=5)
        tk.Button(col, text="Analisar Score", command=self.analisar_score).pack(pady=2)
        tk.Button(col, text="Analisar Cnn", command=self.analise_cnn).pack(pady=2)
        self._btn_sep = tk.Button(col, text="Separar componentes", command=self.separar_componentes)
        self._btn_sep.pack(pady=2)
        tk.Button(col, text="Desclassificar Tudo", command=self.desclassificar_tudo).pack()
        tk.Button(col, text="Limpar classificacoes", command=self.limpar_classificacoes).pack(pady=2)

    def _build_paleta_column(self):
        col = tk.Frame(self.root, bg=BG_COLOR)
        col.grid(row=1, column=1, sticky="n")
        tk.Label(col, text="Ajustes de Composição", font=("Arial", 11, "bold"), bg=BG_COLOR).pack()
        self.paletas = UiFunctions.PaletaMetaDados(self.get_pickcolor, self.on_paleta_change)
        self.paletas.criar_paletas(col)

    def _build_analysis_column(self):
        col = tk.Frame(self.root, bg=BG_COLOR)
        col.grid(row=1, column=2, sticky="n")
        tk.Label(col, text="Analise", font=("Arial", 11, "bold"), bg=BG_COLOR).pack()

        bloco_cats = tk.LabelFrame(col, text="Composição", bg=BG_COLOR, fg="white")
        bloco_cats.pack(fill="both", padx=4, pady=(4, 2))
        self.lbl_g = tk.Label(bloco_cats, text="", bg=BG_COLOR)
        self.lbl_c = tk.Label(bloco_cats, text="", bg=BG_COLOR)
        self.lbl_k = tk.Label(bloco_cats, text="", bg=BG_COLOR)
        self.lbl_g.pack(anchor="w", padx=6, pady=1)
        self.lbl_c.pack(anchor="w", padx=6, pady=1)
        self.lbl_k.pack(anchor="w", padx=6, pady=1)

        bloco_resumo = tk.LabelFrame(col, text="Resumo", bg=BG_COLOR, fg="white")
        bloco_resumo.pack(fill="both", padx=4, pady=(2, 4))
        self.lbl_u = tk.Label(bloco_resumo, text="", bg=BG_COLOR)
        self.lbl_f = tk.Label(bloco_resumo, text="", bg=BG_COLOR)
        self.lbl_visao = tk.Label(bloco_resumo, text="", bg=BG_COLOR)
        self.lbl_u.pack(anchor="w", padx=6, pady=1)
        self.lbl_f.pack(anchor="w", padx=6, pady=1)
        self.lbl_visao.pack(anchor="w", padx=6, pady=1)
        self.lbl_quebra = tk.Label(bloco_resumo, text="", bg=BG_COLOR)
        self.lbl_quebra.pack(anchor="w", padx=6, pady=1)

        bloco_deger = tk.LabelFrame(col, text="Degerminacao", bg=BG_COLOR, fg="white")
        bloco_deger.pack(fill="both", padx=4, pady=(2, 4))
        self.lbl_deger = tk.Label(bloco_deger, text="", bg=BG_COLOR)
        self.lbl_deger.pack(anchor="w", padx=6, pady=1)

        self._reset_analise_labels()

    def _carregar_logo(self):
        caminho_logo = os.path.join(self.base_dir, "imagens", "logos", "LOGO-YRIS.png")
        if not os.path.isfile(caminho_logo):
            return
        try:
            logo = Image.open(caminho_logo)
            w, h = logo.size
            if h > 0:
                nova_altura = 30
                nova_largura = max(1, int(w * (nova_altura / h)))
                logo = logo.resize((nova_largura, nova_altura), Image.LANCZOS)
            self.logo_img = ImageTk.PhotoImage(logo)
            logo_frame = tk.Frame(self.header, bg=BG_COLOR)
            logo_frame.pack()
            tk.Label(logo_frame, image=self.logo_img, bg=BG_COLOR).pack()
        except Exception:
            self.logo_img = None

    def _atualizar_canvas(self):
        if not self.img:
            self.canvas.delete("all")
            return
        rgb = cv2.cvtColor(self.img.matriz_NumPy, cv2.COLOR_BGR2RGB)
        self.navigator.set_image(Image.fromarray(rgb))

    def _reset_analise_labels(self):
        self.lbl_g.config(text="Germen:   -- %")
        self.lbl_c.config(text="Casca:    -- %")
        self.lbl_k.config(text="Canjica:  -- %")
        self.lbl_u.config(text="Util:     -- %")
        self.lbl_f.config(text="Exclusao: -- %")
        self.lbl_deger.config(text="Score --")
        self.lbl_quebra.config(text="Quebra: --")
        self.lbl_visao.config(text="Visao: --")

    def _require_imagem(self):
        if not self.img or getattr(self.img, "matriz_NumPy", None) is None:
            raise ValueError("Nenhuma imagem carregada para processar.")
        return self.img

    def _require_paletas(self):
        if not self.paletas or not self.paletas.refs:
            raise ValueError("Paletas nao carregadas para processar.")
        return self.paletas.refs

    def buscar_imagem(self):
        pasta = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
        pasta.filtrar_arquivos(".png")

        if not pasta.lista_arquivos:
            self.img = None
            self.canvas.delete("all")
            self.TxtDirImagem.config(text="")
            self.TxtQtdItens.config(text="")
            self._reset_analise_labels()
            mb.showinfo("Fim", "Nenhuma imagem nao classificada.")
            return

        caminho = pasta.lista_arquivos[0]
        self.img = ImageFunctions.Imagem(caminho)
        self.pickcolor = None
        self._reset_analise_labels()  # evita exibir resultados da imagem anterior
        self._atualizar_canvas()

        if self.auto_analisar_var.get():
            try:
                self.analisar_imagem()
            except Exception:
                pass

        self.TxtDirImagem.config(text=self.img.nome)
        self.TxtQtdItens.config(text=f"{pasta.quantidade_arquivos} imagens para classificar.")

    def classificar_imagem(self, score_val):
        img = self._require_imagem()
        refs = self._require_paletas()

        registrar_score = self.registrar_score_var.get()

        if registrar_score:
            comp = ProcessImageFunctions.calcular_composicao(img, refs)
            q = ProcessImageFunctions.avaliar_quebra(img)
            visao = ProcessImageFunctions.avaliar_visao(img)
            score_info = ProcessImageFunctions.calcular_score(
                img,
                refs,
                composicao=comp,
                quebra_indice=q.get("score_quebra"),
                visao_indice=visao.get("indice_visao"),
            )

            registro = {
                "arquivo": self.img.nome,
                "caminho": self.img.caminho,
                "score": score_val,
                "metodo_score": score_info.get("metodo"),
                "germen": round(comp.get("germen", 0.0), 4),
                "casca": round(comp.get("casca", 0.0), 4),
                "canjica": round(comp.get("canjica", 0.0), 4),
                "util": round(comp.get("util", 0.0), 4),
                "exclusao": round(comp.get("exclusao", 0.0), 4),
                "quebra_indice": round(q.get("score_quebra", 0.0), 4),
                "classe_quebra": q.get("classe", "indefinido"),
                "indice_visao": round(visao.get("indice_visao", 0.0), 4),
                "classe_visao": visao.get("classe", "indefinido"),
                "data_hora": datetime.now().isoformat(timespec="seconds"),
            }
            FileFunctions.registrar_score(registro)

        pasta = FileFunctions.Pasta(self.dirDataBase, str(score_val))
        if FileFunctions.MoverArquivo(self.img.caminho, pasta.dir).get("ok"):
            self.buscar_imagem()

    def desclassificar_tudo(self):
        pasta_nc = FileFunctions.Pasta(self.dirDataBase, "ImgNotClass")
        destino = pasta_nc.dir

        total = 0
        for i in range(1, 11):
            pasta = FileFunctions.Pasta(self.dirDataBase, str(i))
            for arq in pasta.lista_arquivos:
                if FileFunctions.MoverArquivo(arq, destino).get("ok"):
                    total += 1

        mb.showinfo("OK", f"{total} imagens movidas.")
        self.buscar_imagem()

    def pegar_cor(self, event):
        self._require_imagem()

        coords = self.navigator.canvas_to_image_coords(event.x, event.y)
        if coords is None:
            raise ValueError("Clique deve ocorrer dentro da imagem exibida.")

        x, y = coords
        b, g, r = self.img.matriz_NumPy[y, x]
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        self.pickcolor = hsv

        H, S, V = hsv
        self.paletas.atualizar_cor_preview(UiFunctions.rgb_from_hsv_hex(H, S, V))

    def on_paleta_change(self):
        self._reset_analise_labels()

    def get_pickcolor(self):
        return self.pickcolor

    def separar_componentes(self):
        refs = self._require_paletas()
        caminho = fd.askopenfilename(
            title="Selecione a imagem para separar componentes",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp"), ("Todos", "*.*")]
        )
        if not caminho:
            return

        try:
            img_sel = ImageFunctions.Imagem(caminho)
        except Exception as exc:  # noqa: BLE001
            mb.showerror("Separar componentes", f"Falha ao carregar imagem: {exc}")
            return

        base_nome = os.path.splitext(os.path.basename(caminho))[0]
        pasta_destino = os.path.join(os.path.dirname(caminho), f"componentes_separados_{base_nome}")
        self._btn_sep.config(state="disabled")
        mb.showinfo("Separar componentes", "Processando em segundo plano. Aguarde concluir.")

        def worker():
            salvos = ProcessImageFunctions.separar_componentes_sam(
                img_sel,
                pasta_destino,
                paletas=refs
            )

            def finalizar():
                self._btn_sep.config(state="normal")
                if not salvos:
                    mb.showinfo("Separar componentes", "Nenhuma camada salva.")
                else:
                    mb.showinfo("Separar componentes", f"{len(salvos)} componente(s) salvo(s) em {pasta_destino}.")

            self.root.after(0, finalizar)

        threading.Thread(target=worker, daemon=True).start()

    def analisar_imagem(self):
        self._require_imagem()
        self.analisar_composicao()
        self.analisar_score()
        self.analise_cnn()

    def analisar_composicao(self):
        img = self._require_imagem()
        refs = self._require_paletas()
        r = ProcessImageFunctions.calcular_composicao(img, refs)
        self.lbl_g.config(text=f"Germen:   {r.get('germen', 0):.1f}%")
        self.lbl_c.config(text=f"Casca:    {r.get('casca', 0):.1f}%")
        self.lbl_k.config(text=f"Canjica:  {r.get('canjica', 0):.1f}%")
        self.lbl_u.config(text=f"Util:     {r.get('util', 0):.1f}%")
        self.lbl_f.config(text=f"Exclusao: {r.get('exclusao', 0):.1f}%")

    def analisar_score(self):
        img = self._require_imagem()
        refs = self._require_paletas()
        comp = ProcessImageFunctions.calcular_composicao(img, refs)
        q = ProcessImageFunctions.avaliar_quebra(img)
        visao = ProcessImageFunctions.avaliar_visao(img)
        score_info = ProcessImageFunctions.calcular_score(
            img,
            refs,
            composicao=comp,
            quebra_indice=q.get("score_quebra"),
            visao_indice=visao.get("indice_visao"),
        )
        score_val = score_info.get("score", 0)
        self.lbl_deger.config(text=f"Score {score_val:.2f}")
        self.lbl_visao.config(text=f"Visao: {visao.get('indice_visao', 0):.2f}")

    def analise_cnn(self):
        img = self._require_imagem()
        q = ProcessImageFunctions.avaliar_quebra(img)
        score_q = q.get("score_quebra", 0.0)
        classe = q.get("classe", "indefinido")
        self.lbl_quebra.config(text=f"Quebra: {score_q:.2f} ({classe})")
        visao = ProcessImageFunctions.avaliar_visao(img)
        self.lbl_visao.config(text=f"Visao: {visao.get('indice_visao', 0):.2f}")

    def _voltar_menu(self):
        if self.on_back:
            self.on_back()
        else:
            self.root.destroy()

    def limpar_classificacoes(self):
        """Remove os arquivos de historico de classificacoes (JSON/CSV)."""
        caminhos = [
            FileFunctions.caminho_scores_json(),
            FileFunctions.caminho_scores_csv(),
        ]
        removidos = 0
        for caminho in caminhos:
            try:
                if caminho and os.path.isfile(caminho):
                    os.remove(caminho)
                    removidos += 1
            except Exception:
                pass
        if removidos:
            mb.showinfo("Limpar classificacoes", f"{removidos} arquivo(s) removido(s).")
        else:
            mb.showinfo("Limpar classificacoes", "Nenhum arquivo de classificacao encontrado.")
