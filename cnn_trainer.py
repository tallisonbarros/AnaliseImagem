# ------------------- cnn_trainer.py
"""
Utilitario para:
1) Treinar classificador de quebra (0=integro, 1=parcial, 2=quebrado) e salvar TorchScript (.pt).
2) Rotular rapidamente um dataset de imagens inteiras com essas classes.
"""
import os
import glob
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    nn = None
    Dataset = object  # type: ignore
    DataLoader = None


CLASSES = ["integro", "parcial", "quebrado"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


class _QuebraDataset(Dataset):
    def __init__(self, pasta, tamanho=224):
        self.tamanho = tamanho
        self.samples = self._listar(pasta)

    def _listar(self, pasta):
        samples = []
        if not os.path.isdir(pasta):
            return samples
        for cls in os.listdir(pasta):
            cls_path = os.path.join(pasta, cls)
            if not os.path.isdir(cls_path):
                continue
            cls_norm = cls.lower()
            cls_id = CLASS_TO_ID.get(cls_norm)
            if cls_id is None and cls_norm.isdigit():
                try:
                    cls_id = int(cls_norm)
                except Exception:
                    cls_id = None
            if cls_id is None or cls_id not in (0, 1, 2):
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                for img_path in glob.glob(os.path.join(cls_path, ext)):
                    samples.append((img_path, cls_id))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_id = self.samples[idx]
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Falha ao carregar {img_path}")

        bgr = cv2.resize(bgr, (self.tamanho, self.tamanho), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        img_t = torch.from_numpy(rgb.transpose(2, 0, 1))
        label_t = torch.tensor(cls_id, dtype=torch.long)
        return img_t, label_t


class _ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MiniClassifier(nn.Module):
    def __init__(self, base=16, num_classes=3):
        super().__init__()
        self.down1 = _ConvBlock(3, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = _ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = _ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, base * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base * 4, num_classes),
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.pool1(x)
        x = self.down2(x)
        x = self.pool2(x)
        x = self.down3(x)
        x = self.pool3(x)
        return self.head(x)


def treinar_modelo(pasta_dataset, caminho_saida, epochs=5, batch_size=8, lr=1e-3, tamanho=224, device=None):
    """
    Treina classificador 0=integro, 1=parcial, 2=quebrado e salva TorchScript.
    Estrutura esperada:
      pasta_dataset/
        0/ (ou integro/)
        1/ (ou parcial/)
        2/ (ou quebrado/)
    """
    if not HAS_TORCH:
        return {"ok": False, "erro": "PyTorch nao instalado."}
    if not os.path.isdir(pasta_dataset):
        return {"ok": False, "erro": f"Pasta do dataset invalida: {pasta_dataset}"}

    dataset = _QuebraDataset(pasta_dataset, tamanho=tamanho)
    if len(dataset) == 0:
        return {"ok": False, "erro": "Nenhuma imagem nas pastas 0/1/2 (ou integro/parcial/quebrado)."}

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = MiniClassifier(base=16, num_classes=3).to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss()

    modelo.train()
    for ep in range(epochs):
        perdas = []
        for imgs, labels in dl:
            imgs = imgs.to(dev)
            labels = labels.to(dev)
            opt.zero_grad()
            logits = modelo(imgs)
            loss = criterio(logits, labels)
            loss.backward()
            opt.step()
            perdas.append(float(loss.item()))
        media_loss = float(np.mean(perdas)) if perdas else 0.0
        print(f"[treino] ep {ep+1}/{epochs} loss={media_loss:.4f}")

    modelo.eval()
    exemplo = torch.zeros(1, 3, tamanho, tamanho, device=dev)
    scripted = torch.jit.trace(modelo, exemplo)
    os.makedirs(os.path.dirname(caminho_saida) or ".", exist_ok=True)
    scripted.save(caminho_saida)

    return {"ok": True, "modelo": caminho_saida, "amostras": len(dataset), "epochs": epochs, "device": dev}


# -------- Ferramenta de construcao de dataset (rotulo 0/1/2) --------

class ImageLabeler:
    """
    Rotulador simples para classificacao 0/1/2.
    Controles:
      - Teclas '0', '1', '2' para rotular e salvar em subpasta destino (0/1/2).
      - Tecla 'n' para pular.
      - Tecla 'q' ou ESC para sair.
    """
    def __init__(self, pasta_origem, pasta_destino, tamanho=224):
        self.pasta_origem = pasta_origem
        self.pasta_destino = pasta_destino
        self.tamanho = tamanho
        self.imagens = self._listar_imagens(pasta_origem)
        self.idx = 0
        self.img_view = None

    def _listar_imagens(self, pasta):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(pasta, ext)))
        files.sort()
        return files

    def _carregar_atual(self):
        if self.idx >= len(self.imagens):
            return False
        path = self.imagens[self.idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return False
        bgr = cv2.resize(bgr, (self.tamanho, self.tamanho), interpolation=cv2.INTER_AREA)
        self.img_view = bgr
        return True

    def _salvar(self, img_path, label_id):
        base = os.path.splitext(os.path.basename(img_path))[0]
        destino_cls = os.path.join(self.pasta_destino, str(label_id))
        os.makedirs(destino_cls, exist_ok=True)
        out_img = os.path.join(destino_cls, f"{base}.png")
        cv2.imwrite(out_img, self.img_view)
        print(f"[dataset] salvo: {out_img} -> classe {label_id}")

    def run(self):
        if not self.imagens:
            print("[dataset] Nenhuma imagem encontrada.")
            return
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        while self.idx < len(self.imagens):
            ok = self._carregar_atual()
            if not ok:
                self.idx += 1
                continue
            path = self.imagens[self.idx]
            while True:
                cv2.imshow("img", self.img_view)
                key = cv2.waitKey(50) & 0xFF
                if key in (27, ord('q')):
                    cv2.destroyAllWindows()
                    return
                if key == ord('n'):
                    break
                if key in (ord('0'), ord('1'), ord('2')):
                    label_id = int(chr(key))
                    self._salvar(path, label_id)
                    break
            self.idx += 1
        cv2.destroyAllWindows()


def construir_dataset(pasta_origem, pasta_destino, tamanho=224):
    """
    Abre rotulador simples para classificar imagens inteiras em 0/1/2 e salvar no dataset.
    """
    if not os.path.isdir(pasta_origem):
        print(f"[dataset] Pasta de origem invalida: {pasta_origem}")
        return
    annot = ImageLabeler(pasta_origem, pasta_destino, tamanho=tamanho)
    annot.run()
