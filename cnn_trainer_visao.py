# ------------------- cnn_trainer_visao.py
"""
Treino do classificador de visao (notas 1..10) e salvamento em TorchScript (.pt).
Estrutura esperada:
  dataset_visao/
    1/
    2/
    ...
    10/
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


CLASSES = [str(i) for i in range(1, 11)]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


class _VisaoDataset(Dataset):
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
            cls_norm = cls.strip()
            cls_id = CLASS_TO_ID.get(cls_norm)
            if cls_id is None and cls_norm.isdigit():
                try:
                    val = int(cls_norm)
                    if 1 <= val <= 10:
                        cls_id = val - 1
                except Exception:
                    cls_id = None
            if cls_id is None:
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
    def __init__(self, base=16, num_classes=10):
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
    if not HAS_TORCH:
        return {"ok": False, "erro": "PyTorch nao instalado."}
    if not os.path.isdir(pasta_dataset):
        return {"ok": False, "erro": f"Pasta do dataset invalida: {pasta_dataset}"}

    dataset = _VisaoDataset(pasta_dataset, tamanho=tamanho)
    if len(dataset) == 0:
        return {"ok": False, "erro": "Nenhuma imagem nas pastas 1..10."}

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = MiniClassifier(base=16, num_classes=10).to(dev)
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
        print(f"[treino visao] ep {ep+1}/{epochs} loss={media_loss:.4f}")

    modelo.eval()
    exemplo = torch.zeros(1, 3, tamanho, tamanho, device=dev)
    scripted = torch.jit.trace(modelo, exemplo)
    os.makedirs(os.path.dirname(caminho_saida) or ".", exist_ok=True)
    scripted.save(caminho_saida)

    return {"ok": True, "modelo": caminho_saida, "amostras": len(dataset), "epochs": epochs, "device": dev}
