# ------------------- UiFunctions.py
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import colorsys


def rgb_from_hsv_hex(H, S, V):
    r, g, b = colorsys.hsv_to_rgb(H / 179.0, S / 255.0, V / 255.0)
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def exibir_canvas(canvas, pil_image, zoom, pan_x, pan_y):
    largura = int(pil_image.width * zoom)
    altura = int(pil_image.height * zoom)

    img_zoom = pil_image.resize((largura, altura))
    img_tk = ImageTk.PhotoImage(img_zoom)

    canvas.delete("all")
    canvas.create_image(pan_x, pan_y, anchor="nw", image=img_tk)

    return img_tk


class PaletaCategoriaCores:
    def __init__(self, get_pickcolor_func, on_change=None):
        self.get_pickcolor = get_pickcolor_func
        self.on_change = on_change
        self.refs = {}
        self.HSVcor = None

    def criar_paletas(self, frame_pai):
        bloco = tk.LabelFrame(frame_pai, text="Categorias analisadas", font=("Arial", 10, "bold"))
        bloco.pack(pady=10)

        categorias = ["Germen", "Casca", "Canjica"]
        self.refs = {}

        for nome in categorias:
            nome_low = nome.lower()

            linha = tk.Frame(bloco)
            linha.pack(anchor="w", pady=3)

            lbl = tk.Label(linha, text=nome, font=("Arial", 9))
            lbl.pack(side="left", padx=5)

            frame_cores = tk.Frame(linha)
            frame_cores.pack(side="left", padx=5)

            btn_add = tk.Button(
                linha,
                text="+",
                width=2,
                command=lambda n=nome_low: self.cadastrar_cor(n)
            )
            btn_add.pack(side="left", padx=2)

            btn_del = tk.Button(
                linha,
                text="-",
                width=2,
                command=lambda n=nome_low: self.remover_ultima_cor(n)
            )
            btn_del.pack(side="left", padx=2)

            self.refs[nome_low] = {
                "cores": [],
                "frame_cores": frame_cores
            }

        frame_pick = tk.Frame(bloco)
        frame_pick.pack(pady=5)

        self.HSVcor = tk.Label(frame_pick, width=3, height=1, relief="solid", bg="#000000")
        self.HSVcor.pack()

        return self.refs

    def cadastrar_cor(self, categoria):
        hsv = self.get_pickcolor()
        if hsv is None:
            return

        H, S, V = map(int, hsv)
        self.refs[categoria]["cores"].append((H, S, V))

        cor_hex = rgb_from_hsv_hex(H, S, V)
        tk.Label(
            self.refs[categoria]["frame_cores"],
            bg=cor_hex, width=2, height=1, relief="ridge"
        ).pack(side="left", padx=1)

        if self.on_change:
            self.on_change()

    def remover_ultima_cor(self, categoria):
        lista = self.refs[categoria]["cores"]
        if not lista:
            return

        lista.pop()

        frame = self.refs[categoria]["frame_cores"]
        filhos = frame.winfo_children()
        if filhos:
            filhos[-1].destroy()

        if self.on_change:
            self.on_change()


class CanvasNavigator:
    def __init__(self, canvas):
        self.canvas = canvas
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.img_original = None   # PIL.Image
        self.img_tk = None         # ImageTk.PhotoImage

        # posição inicial do pan (evita AttributeError)
        self._px = 0
        self._py = 0

        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<ButtonPress-1>", self._pan_start, add="+")
        self.canvas.bind("<B1-Motion>", self._pan_move, add="+")

    def set_image(self, pil_image):
        self.img_original = pil_image

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih = pil_image.size

        escala = min(cw / iw, ch / ih, 1.0)

        self.zoom = escala
        self.pan_x = 0
        self.pan_y = 0
        self.min_zoom = escala

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
            self.zoom *= 1.1
        else:
            novo_zoom = self.zoom / 1.1
            if novo_zoom < self.min_zoom:
                self.zoom = self.min_zoom
            else:
                self.zoom = novo_zoom

        self._redraw()

    def _pan_start(self, event):
        self._px = event.x
        self._py = event.y

    def _pan_move(self, event):
        if self.img_original is None:
            return

        dx = event.x - self._px
        dy = event.y - self._py

        self.pan_x += dx
        self.pan_y += dy

        self._px = event.x
        self._py = event.y

        self._redraw()

    def canvas_to_image_coords(self, x_canvas, y_canvas):
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
