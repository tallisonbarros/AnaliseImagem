# UiFunctions.py
import cv2
import tkinter as tk
import tkinter.messagebox as mb
from PIL import Image, ImageTk
import colorsys
import numpy as np

def exibir_imagem_ui(imagem_numpy, label_destino):
    """
    Recebe a matriz NumPy da imagem (BGR) e um tk.Label onde
    a imagem será exibida.
    """
    # converter BGR → RGB
    img_rgb = cv2.cvtColor(imagem_numpy, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.resize((350, 350))

    imagem_tk = ImageTk.PhotoImage(pil_img)

    # guardar referência na própria label para não perder a imagem
    label_destino.image = imagem_tk
    label_destino.config(image=imagem_tk)


def criar_inputs_hsv(frame, titulo, valores_padrao=None):
    bloco = tk.LabelFrame(frame, text=titulo)
    bloco.pack(pady=5)

    inputs = {}

    for componente in ["H", "S", "V"]:
        linha = tk.Frame(bloco)
        linha.pack()

        tk.Label(linha, text=componente + "_min").pack(side="left")
        entry_min = tk.Entry(linha, width=5)
        entry_min.pack(side="left")

        tk.Label(linha, text=componente + "_max").pack(side="left")
        entry_max = tk.Entry(linha, width=5)
        entry_max.pack(side="left")

        # ✔️ Preenche valores padrão, se fornecidos
        if valores_padrao:
            vmin, vmax = valores_padrao.get(componente, (0, 255))
            entry_min.insert(0, str(vmin))
            entry_max.insert(0, str(vmax))

        inputs[componente] = (entry_min, entry_max)

    return inputs


def coletar_hsv(paletas):
    # tolerâncias em torno de cada ponto HSV cadastrado
    DELTA_H = 5
    DELTA_S = 40
    DELTA_V = 40

    resultado = {}

    for categoria, dados in paletas.items():
        # ignorar o label HSVCor
        if categoria == "HSVCor":
            continue

        # categorias válidas têm o campo "cores"
        if "cores" not in dados:
            continue

        cores = dados["cores"]
        if not cores:
            continue

        faixas = []

        for (H, S, V) in cores:
            H = int(H)
            S = int(S)
            V = int(V)

            h_min = max(H - DELTA_H, 0)
            h_max = min(H + DELTA_H, 179)

            s_min = max(S - DELTA_S, 0)
            s_max = min(S + DELTA_S, 255)

            v_min = max(V - DELTA_V, 0)
            v_max = min(V + DELTA_V, 255)

            faixas.append({
                "min": (h_min, s_min, v_min),
                "max": (h_max, s_max, v_max),
            })


        # só adiciona se tiver pelo menos 1 faixa
        if faixas:
            resultado[categoria] = faixas

    return resultado


def exibir_canvas(canvas, pil_image, zoom, pan_x, pan_y):
    largura = int(pil_image.width * zoom)
    altura  = int(pil_image.height * zoom)

    img_zoom = pil_image.resize((largura, altura))
    img_tk = ImageTk.PhotoImage(img_zoom)

    canvas.delete("all")
    canvas.create_image(pan_x, pan_y, anchor="nw", image=img_tk)

    return img_tk


# ==========================
# HELPER PALETA CATEGORIA CORES
# ==========================
class PaletaCategoriaCores:
    def __init__(self, get_pickcolor_func, on_change=None):
        self.refs = {}
        self.get_pickcolor = get_pickcolor_func
        self.HSVcor = None
        self.on_change = on_change 

    def criar_paletas(self, frame_pai):
        bloco = tk.LabelFrame(frame_pai, text="Categorias analisadas", font=("Arial", 10, "bold"))
        bloco.pack(pady=10)

        categorias = ["Germen", "Casca", "Canjica"]
        self.refs = {}

        for nome in categorias:
            linha = tk.Frame(bloco)
            linha.pack(anchor="w", pady=3)

            # Label com o nome da categoria
            lbl = tk.Label(linha, text=nome, font=("Arial", 9))
            lbl.pack(side="left", padx=5)

            frame_cores = tk.Frame(linha)
            frame_cores.pack(side="left", padx=5)

            # Botão "+": adicionar cor
            btn_add = tk.Button(
                linha,
                text="+",
                font=("Arial", 8, "bold"),
                width=2,
                height=1,
                relief="groove",
                bg="#e0e0e0",
                command=lambda nome=nome.lower(): self.cadastrar_cor(nome)
            )
            btn_add.pack(side="left", padx=3)

            # Botão "-": remover última cor
            btn_del = tk.Button(
                linha,
                text="-",
                font=("Arial", 8, "bold"),
                width=2,
                height=1,
                relief="groove",
                bg="#ffcccc",
                command=lambda nome=nome.lower(): self.remover_ultima_cor(nome)
            )
            btn_del.pack(side="left", padx=3)

            # Guarda referências
            self.refs[nome.lower()] = {
                "label": lbl,
                "botao_add": btn_add,
                "botao_del": btn_del,
                "cores": [],
                "frame_cores": frame_cores
            }

        # Frame para a exibição da cor atual
        frame_pick = tk.Frame(bloco)
        frame_pick.pack(pady=5)
        # cor atual
        self.HSVcor = tk.Label(frame_pick, width=3, height=1, relief="solid", bg="#000000")
        self.HSVcor.pack()

        return self.refs
    
    def cadastrar_cor(self, categoria):
        hsv = self.get_pickcolor()

        if hsv is None:
            mb.showerror("Erro", "Nenhuma cor foi capturada ainda.")
            return
        
        H, S, V = hsv

        self.refs[categoria]["cores"].append((H, S, V))
        self._adicionar_cor(self.refs[categoria]["frame_cores"], (H, S, V))

        mb.showinfo("Cor cadastrada", f"Categoria: {categoria}\nHSV: ({H}, {S}, {V})")

        if self.on_change is not None:
            self.on_change()

    def _adicionar_cor(self, frame, hsv):
        # Converte HSV → RGB para mostrar a cor correta

        h, s, v = hsv
        r, g, b = colorsys.hsv_to_rgb(h/179, s/255, v/255)
        r = int(r*255)
        g = int(g*255)
        b = int(b*255)

        cor_hex = f"#{r:02x}{g:02x}{b:02x}"

        quad = tk.Label(frame, bg=cor_hex, width=2, height=1, relief="ridge")
        quad.pack(side="left", padx=2)

    def remover_ultima_cor(self, categoria):
        lista = self.refs[categoria]["cores"]
        frame = self.refs[categoria]["frame_cores"]

        if not lista:
            mb.showinfo("Aviso", "Nenhuma cor para remover.")
            return

        # Remove da lista interna
        lista.pop()

        # Remove o último quadradinho do frame
        filhos = frame.winfo_children()
        if filhos:
            filhos[-1].destroy()

        mb.showinfo("Cor removida", f"Última cor da categoria '{categoria}' foi removida.")

        if self.on_change is not None:
            self.on_change()



# ==========================
# HELPER DE NAVEGAÇÃO
# ==========================
class CanvasNavigator:
    def __init__(self, canvas):
        self.canvas = canvas
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_init_x = 0
        self.pan_init_y = 0

        self.img_original = None   # PIL.Image
        self.img_tk = None         # ImageTk.PhotoImage

        # binds básicos
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<ButtonPress-1>", self._pan_start, add="+")
        self.canvas.bind("<B1-Motion>", self._pan_move, add="+")

    def set_image(self, pil_image):
        """Define nova imagem e reseta zoom/pan."""
        self.img_original = pil_image  # <- guarda a imagem!

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        iw, ih = pil_image.size

        # calcula escala para caber no canvas, sem aumentar
        escala = min(cw / iw, ch / ih, 1.0)

        self.zoom = escala
        self.pan_x = 0
        self.pan_y = 0
        self.min_zoom = escala      # <<-- SALVA O LIMITE
        self._redraw()

    def _redraw(self):
        if self.img_original is None:
            return
        self.img_tk = exibir_canvas(
            self.canvas,
            self.img_original,
            self.zoom,
            self.pan_x,
            self.pan_y
        )

    def _on_scroll(self, event):
        if self.img_original is None:
            return

        if event.delta > 0:
            # zoom in (sempre permitido)
            self.zoom *= 1.1
        else:
            # zoom out → respeita min_zoom
            novo_zoom = self.zoom / 1.1
            if novo_zoom < self.min_zoom:
                self.zoom = self.min_zoom
            else:
                self.zoom = novo_zoom

        self._redraw()


    def _pan_start(self, event):
        self.pan_init_x = event.x
        self.pan_init_y = event.y

    def _pan_move(self, event):
        if self.img_original is None:
            return


        dx = event.x - self.pan_init_x
        dy = event.y - self.pan_init_y

        self.pan_x += dx
        self.pan_y += dy

        self.pan_init_x = event.x
        self.pan_init_y = event.y

        self._redraw()

    def canvas_to_image_coords(self, x_canvas, y_canvas):
        """Converte coordenadas do canvas para coordenadas da imagem original."""
        if self.img_original is None:
            return None

        x_zoom = x_canvas - self.pan_x
        y_zoom = y_canvas - self.pan_y

        if x_zoom < 0 or y_zoom < 0:
            return None

        if self.zoom == 0:
            return None

        x = int(x_zoom / self.zoom)
        y = int(y_zoom / self.zoom)

        if x < 0 or y < 0 or x >= self.img_original.width or y >= self.img_original.height:
            return None

        return x, y

def criar_input_preprocess(frame_pai):

    bloco = tk.LabelFrame(frame_pai, text="Pré-Processamento", font=("Arial", 10, "bold"))
    bloco.pack(side="right", padx=10, pady=10)

    # ===== VALORES PADRÃO =====

    hsv_germen = criar_inputs_hsv(
        bloco, "Germen - Azul",
        valores_padrao={
            "H": (99, 99),     # azul OpenCV
            "S": (255, 255),
            "V": (232, 232)
        }
    )

    hsv_casca = criar_inputs_hsv(
        bloco, "Casca - Amarelo",
        valores_padrao={
            "H": (28, 28),      # amarelo OpenCV
            "S": (255, 255),
            "V": (255, 255)
        }
    )

    hsv_canjica = criar_inputs_hsv(
        bloco, "Canjica - Preto",
        valores_padrao={
            "H": (0, 0),      # preto não tem H, então deixa amplo
            "S": (0, 0),
            "V": (0, 0)
        }
    )

    HSVcor = tk.Label(bloco, text="Cor:", font=("Arial", 8, "bold"))
    HSVcor.pack()

    preprocess = {
        "germen": hsv_germen,
        "casca": hsv_casca,
        "canjica": hsv_canjica,
        "HSVCor": HSVcor
    }
    
    return preprocess



def extrair_cores_nao_categorizadas(imagem_bgr, paletas_refs, max_cores=60):
    if imagem_bgr is None:
        return []

    # 1) Pega todas as faixas já cadastradas (usa a mesma lógica do coletar_hsv)
    faixas_por_categoria = coletar_hsv(paletas_refs)
    faixas = []
    for lista in faixas_por_categoria.values():
        faixas.extend(lista)

    # 2) Reduz imagem para algo pequeno (ex.: 40x40) para não explodir em cores
    h, w = imagem_bgr.shape[:2]
    alvo = 40
    escala = min(alvo / max(h, w), 1.0)
    if escala < 1.0:
        nova_larg = int(w * escala)
        nova_alt = int(h * escala)
        img_small = cv2.resize(imagem_bgr, (nova_larg, nova_alt), interpolation=cv2.INTER_AREA)
    else:
        img_small = imagem_bgr.copy()

    # 3) Converte para HSV
    img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    # 4) Quantiza as cores para evitar milhões de variações
    pixels = img_hsv.reshape(-1, 3)
    cores_quantizadas = {}

    for (H, S, V) in pixels:
        # arredonda para "baldes" (ajusta se quiser mais fino/grosso)
        key = (
            int(H // 5) * 5,     # passo de 5 em H
            int(S // 32) * 32,   # passo de 32 em S
            int(V // 32) * 32    # passo de 32 em V
        )
        # guarda um exemplo real desse balde
        if key not in cores_quantizadas:
            cores_quantizadas[key] = (int(H), int(S), int(V))

    candidatos = list(cores_quantizadas.values())

    # 5) Filtra: remove cores que caem em alguma faixa cadastrada
    cores_livres = []

    for (H, S, V) in candidatos:
        dentro_de_alguma_faixa = False
        for faixa in faixas:
            h_min, s_min, v_min = faixa["min"]
            h_max, s_max, v_max = faixa["max"]

            if (h_min <= H <= h_max and
                s_min <= S <= s_max and
                v_min <= V <= v_max):
                dentro_de_alguma_faixa = True
                break

        if not dentro_de_alguma_faixa:
            cores_livres.append((H, S, V))

        if len(cores_livres) >= max_cores:
            break

    return cores_livres
