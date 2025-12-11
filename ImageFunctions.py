# ------------------- ImageFunctions.py
import os
import cv2

class Imagem:
    def __init__(self, caminho):
        self.caminho = caminho
        self.matriz_NumPy = None
        self.hsv = None      # <<< ADICIONADO
        self.altura = None
        self.largura = None
        self.canais = None
        self.total_pixels = None
        self.dtype = None
        self.tamanho_bytes = None
        self.nome = os.path.basename(caminho)
        self.extensao = os.path.splitext(caminho)[1].lower()
        self.diretorio = os.path.dirname(caminho)
        self.valida = False
        self.erro = None

        self._carregar_informacoes()

    def _carregar_informacoes(self):
        if not os.path.isfile(self.caminho):
            self.erro = f"[ERRO] Arquivo não encontrado: {self.caminho}"
            return

        self.tamanho_bytes = os.path.getsize(self.caminho)

        img = cv2.imread(self.caminho)
        if img is None:
            self.erro = f"[ERRO] Não foi possível carregar: {self.caminho}"
            return

        self.matriz_NumPy = img
        self.valida = True

        self.altura, self.largura = img.shape[:2]
        self.canais = img.shape[2] if len(img.shape) == 3 else 1
        self.total_pixels = self.altura * self.largura
        self.dtype = img.dtype

        # <<< AQUI: conversão única e eficiente
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def exibir(self, titulo=None):
        if not self.valida:
            print(self.erro)
            return
        cv2.imshow(titulo or self.nome, self.matriz_NumPy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def qtd_pixels(self):
        return self.matriz_NumPy.shape[0] * self.matriz_NumPy.shape[1]
