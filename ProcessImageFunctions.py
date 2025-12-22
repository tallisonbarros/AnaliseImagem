# ------------------- ProcessImageFunctions.py
import os
import cv2
import ImageFunctions
import FileFunctions
import tkinter.messagebox as mb
import numpy as np

# Integracao SAM1 (Segment Anything) com SamAutomaticMaskGenerator
try:
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    HAS_SAM = True
except Exception:
    HAS_SAM = False
    sam_model_registry = None
    SamAutomaticMaskGenerator = None
    torch = None

# CNN classificador de quebra (integro/parcial/quebrado)
try:
    import cnn_quebra_clf
    HAS_CNN_QUEBRA = True
except Exception as exc:
    HAS_CNN_QUEBRA = False
    cnn_quebra_clf = None
    raise RuntimeError(f"Falha ao importar cnn_quebra_clf: {exc}")

# CNN classificador de visao (nota 1-10)
try:
    import cnn_visao_clf
    HAS_CNN_VISAO = True
except Exception as exc:
    HAS_CNN_VISAO = False
    cnn_visao_clf = None
    raise RuntimeError(f"Falha ao importar cnn_visao_clf: {exc}")


# Configuracao SAM (AMG)
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", os.path.join(os.getcwd(), "sam_vit_h_4b8939.pth"))
SAM_AMG_CFG = {
    # filtros mais rigorosos para evitar pegar a bandeja/fundo
    "points_per_side": 24,
    "pred_iou_thresh": 0.95,
    "stability_score_thresh": 0.97,
    "box_nms_thresh": 0.35,
    "crop_n_layers": 1,
    "min_mask_region_area": 1500,
}

# cache global do predictor
_SAM_MODEL = None
_SAM_AMG = None
PESOS_DEFAULT = {"germen": 1.0, "casca": 1.0, "canjica": 1.0, "quebra": 1.0, "visao": 1.0}


def contar_pixels_hsv(Imagem, hsv_min, hsv_max):
    """Conta quantos pixels estao entre min/max usando HSV ja pre-calculado."""
    mascara = cv2.inRange(Imagem.hsv, hsv_min, hsv_max)
    pixels = cv2.countNonZero(mascara)
    return pixels


def gerar_faixas_por_paleta(paletas, tolerancia=0.10):
    faixas = {}
    for categoria, dados in paletas.items():
        if categoria == "HSVCor":
            continue
        if "cores" not in dados:
            continue
        cores = dados["cores"]
        if not cores:
            continue
        lista_faixas = []
        for (H, S, V) in cores:
            h_delta = int(H * tolerancia)
            s_delta = int(S * tolerancia)
            v_delta = int(V * tolerancia)
            h_min = max(H - h_delta, 0)
            h_max = min(H + h_delta, 179)
            s_min = max(S - s_delta, 0)
            s_max = min(S + s_delta, 255)
            v_min = max(V - v_delta, 0)
            v_max = min(V + v_delta, 255)
            lista_faixas.append({"min": (h_min, s_min, v_min), "max": (h_max, s_max, v_max)})
        faixas[categoria] = lista_faixas
    return faixas


def calcular_percentuais_hsv(img, faixas):
    resultados = {}
    total = img.total_pixels
    for categoria, lista_faixas in faixas.items():
        soma = 0
        for faixa in lista_faixas:
            soma += contar_pixels_hsv(img, faixa["min"], faixa["max"])
        resultados[categoria] = (soma / total) * 100.0
    return resultados


def montar_faixas_por_categoria(paletas, categorias, tolerancia=0.10):
    """Retorna faixas HSV apenas das categorias desejadas."""
    if not paletas:
        return {}
    subset = {k: v for k, v in paletas.items() if k.lower() in categorias}
    return gerar_faixas_por_paleta(subset, tolerancia=tolerancia)


def gerar_mascara_por_faixas(hsv_img, faixas):
    """Combina faixas HSV em uma unica mascara binaria."""
    if not faixas:
        return None
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for faixa in faixas:
        m = cv2.inRange(hsv_img, np.array(faixa["min"]), np.array(faixa["max"]))
        mask = cv2.bitwise_or(mask, m)
    return mask


def calcular_composicao(img, paletas):
    """
    Classifica pixels via KNN no espaco HSV usando as cores de referencia das paletas.
    Retorna percentuais por categoria; 'exclusao' conta V=0 diretamente.
    """
    if not paletas:
        raise ValueError("Paletas nao fornecidas para calcular composicao.")
    if img is None or getattr(img, "matriz_NumPy", None) is None:
        raise ValueError("Imagem invalida para calcular composicao.")

    hsv_img = getattr(img, "hsv", None)
    if hsv_img is None:
        try:
            hsv_img = cv2.cvtColor(img.matriz_NumPy, cv2.COLOR_BGR2HSV)
            img.hsv = hsv_img
        except Exception:
            raise RuntimeError("Falha ao converter imagem para HSV.")

    # Normaliza categorias e junta refs
    categorias_ordem = []
    exemplos = []
    labels = []
    for categoria, dados in paletas.items():
        if not categoria:
            continue
        cat = categoria.lower()
        if cat in ("fundo", "exclusao"):
            cat = "exclusao"
        if cat == "hsvcor":
            continue
        cores = list((dados or {}).get("cores", []))
        if not cores and cat == "exclusao":
            cores = [(0, 0, 0)]  # referencia de fundo
        if not cores:
            continue
        if cat not in categorias_ordem:
            categorias_ordem.append(cat)
        cid = categorias_ordem.index(cat)
        for (H, S, V) in cores:
            exemplos.append((int(H), int(S), int(V)))
            labels.append(cid)

    if "exclusao" not in categorias_ordem:
        categorias_ordem.append("exclusao")

    base_resultados = {cat: 0.0 for cat in categorias_ordem}

    pixels = hsv_img.reshape(-1, 3).astype(np.float32)
    total_pixels = len(pixels)
    if total_pixels == 0:
        raise RuntimeError("Imagem sem pixels validos para composicao.")

    mask_validos = pixels[:, 2] > 0  # V>0 = nao-fundo
    pixels_validos = pixels[mask_validos]
    zeros_v = total_pixels - len(pixels_validos)

    if not exemplos:
        resultados = dict(base_resultados)
        resultados["exclusao"] = (zeros_v / total_pixels) * 100.0
        resultados["util"] = 100.0 - resultados["exclusao"]
        return resultados

    exemplos_arr = np.asarray(exemplos, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)
    k = min(5, len(exemplos_arr))

    bloco = 200000  # processa em blocos para economizar memoria
    contagem = np.zeros(len(categorias_ordem), dtype=np.int64)

    ref_h = exemplos_arr[:, 0][None, :]
    ref_s = exemplos_arr[:, 1][None, :]
    ref_v = exemplos_arr[:, 2][None, :]

    for inicio in range(0, len(pixels_validos), bloco):
        fim = min(inicio + bloco, len(pixels_validos))
        chunk = pixels_validos[inicio:fim]

        dh = np.abs(chunk[:, [0]] - ref_h)
        dh = np.minimum(dh, 180 - dh) / 180.0  # circular em H
        ds = np.abs(chunk[:, [1]] - ref_s) / 255.0
        dv = np.abs(chunk[:, [2]] - ref_v) / 255.0
        dist = dh * dh + ds * ds + dv * dv

        viz_idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        viz_dists = np.take_along_axis(dist, viz_idx, axis=1)
        viz_labels = labels_arr[viz_idx]

        # pesos inversamente proporcionais a distancia (zero -> peso maximo)
        pesos_viz = 1.0 / (np.sqrt(viz_dists) + 1e-6)

        votos = np.zeros((len(chunk), len(categorias_ordem)), dtype=np.float32)
        for cid in range(len(categorias_ordem)):
            votos[:, cid] = np.sum(pesos_viz * (viz_labels == cid), axis=1)

        vencedores = np.argmax(votos, axis=1)
        contagem += np.bincount(vencedores, minlength=len(categorias_ordem))

    idx_exc = categorias_ordem.index("exclusao")
    cont_exclusao_knn = contagem[idx_exc] if idx_exc is not None else 0
    cont_exclusao_total = cont_exclusao_knn + zeros_v
    total_util = max(total_pixels - cont_exclusao_total, 0)

    resultados = dict(base_resultados)
    resultados["exclusao"] = (cont_exclusao_total / total_pixels) * 100.0
    resultados["util"] = 100.0 - resultados["exclusao"]

    for cid, nome_cat in enumerate(categorias_ordem):
        if nome_cat == "exclusao":
            continue
        resultados[nome_cat] = 0.0 if total_util == 0 else (contagem[cid] / total_util) * 100.0

    return resultados


def calcular_score(img, paletas, composicao=None, pesos=None, quebra_indice=None, visao_indice=None):
    """
    Camada 2 - modelo estatistico ponderado para estimar score/nota a partir da composicao.
    Usa regressao linear sobre o historico salvo (Classificacoes.json). Se faltar base, usa pesos default.
    Opcionalmente incorpora o indice de quebra (0-1) como feature adicional.
    """
    if paletas is None:
        raise ValueError("Paletas nao fornecidas para calcular score.")
    comp = composicao or calcular_composicao(img, paletas)

    base = FileFunctions.ler_json(FileFunctions.caminho_scores_json(), []) or []
    X = []
    y = []
    for item in base:
        if not isinstance(item, dict):
            continue
        score = item.get("score", item.get("nota"))
        if score is None:
            continue
        X.append([
            float(item.get("germen", 0.0)),
            float(item.get("casca", 0.0)),
            float(item.get("canjica", 0.0)),
            float(item.get("util", 0.0)),
        ])
        if quebra_indice is not None:
            X[-1].append(float(item.get("quebra_indice", item.get("score_quebra", 0.0))) * 100.0)
        if visao_indice is not None:
            X[-1].append(float(item.get("visao_indice", item.get("indice_visao", 0.0))) * 10.0)
        y.append(float(score))

    feat = np.array([
        float(comp.get("germen", 0.0)),
        float(comp.get("casca", 0.0)),
        float(comp.get("canjica", 0.0)),
        float(comp.get("util", 0.0)),
    ], dtype=np.float64)
    if quebra_indice is not None:
        feat = np.append(feat, float(quebra_indice) * 100.0)
    if visao_indice is not None:
        feat = np.append(feat, float(visao_indice) * 10.0)

    # pesos opcionais para ajustar contribuicao (fallback)
    pesos_default = {"germen": 1.0, "casca": 1.0, "canjica": 1.0, "quebra": 1.0, "visao": 1.0}
    pesos = pesos or pesos_default

    if len(X) >= 2:
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64)
        # regressao linear com termo de bias
        X_design = np.c_[X_mat, np.ones(len(X_mat))]
        coef, _, _, _ = np.linalg.lstsq(X_design, y_vec, rcond=None)
        pred = float(np.dot(np.append(feat, 1.0), coef))
        min_y = float(np.min(y_vec))
        max_y = float(np.max(y_vec))
        if max_y > min_y:
            pred = float(np.clip(pred, min_y, max_y))
        metodo = "regressao"
    else:
        # fallback: media ponderada normalizada (0-10) com pesos ajustaveis
        soma_pesos = sum(pesos.values()) or 1.0
        num = (
            comp.get("germen", 0.0) * pesos.get("germen", 0.0) +
            comp.get("casca", 0.0) * pesos.get("casca", 0.0) +
            comp.get("canjica", 0.0) * pesos.get("canjica", 0.0) +
            (float(quebra_indice) * 100.0 if quebra_indice is not None else 0.0) * pesos.get("quebra", 0.0) +
            (float(visao_indice) * 10.0 if visao_indice is not None else 0.0) * pesos.get("visao", 0.0)
        )
        media = num / soma_pesos
        pred_comp = (media / 100.0) * 10.0
        pred = pred_comp
        metodo = "ponderado"

    return {
        "score": pred,
        "metodo": metodo,
        "composicao": comp,
        "quebra_indice": quebra_indice,
        "visao_indice": visao_indice
    }


def avaliar_visao(img: ImageFunctions.Imagem):
    """
    Avalia nota/indice de visao (1-10) via CNN dedicada.
    """
    if img is None or getattr(img, "matriz_NumPy", None) is None:
        raise ValueError("Imagem invalida para avaliacao de visao.")
    if not (HAS_CNN_VISAO and cnn_visao_clf is not None):
        raise RuntimeError("cnn_visao_clf nao carregado.")

    res = cnn_visao_clf.classificar_visao(img.matriz_NumPy)
    if not res:
        raise RuntimeError("Retorno vazio de cnn_visao_clf.")
    probs = res.get("probs") or []
    indice = float(res.get("indice_visao", 0.0) or res.get("indice", 0.0) or res.get("score", 0.0))
    if not indice and probs:
        pesos = np.arange(1, len(probs) + 1, dtype=np.float32)
        soma_probs = float(np.sum(probs))
        indice = float(np.dot(pesos, probs) / soma_probs) if soma_probs > 0 else float(np.argmax(probs) + 1)
    classe = res.get("classe", str(int(round(indice))) if indice else "indefinido")
    return {"indice_visao": indice, "classe": classe, "probs": probs}


def avaliar_quebra(img: ImageFunctions.Imagem):
    """
    Avalia grau de quebra usando o classificador CNN (integro/parcial/quebrado).
    Retorna score (0-1) e classe textual; depende do modelo CNN.
    """
    if img is None or getattr(img, "matriz_NumPy", None) is None:
        raise ValueError("Imagem invalida para avaliacao de quebra.")
    if not (HAS_CNN_QUEBRA and cnn_quebra_clf is not None):
        raise RuntimeError("cnn_quebra_clf nao carregado.")

    mask = gerar_mascara_fg(img.matriz_NumPy)
    if mask is None:
        raise RuntimeError("Falha ao gerar mascara de foreground.")

    # mantem apenas o maior componente como foreground principal
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("Foreground insuficiente para avaliar quebra.")
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(1 + np.argmax(areas))
    mask_fg = np.zeros_like(mask)
    mask_fg[labels == largest_idx] = 255

    resultado_cnn = cnn_quebra_clf.classificar_quebra(img.matriz_NumPy, mask_fg=mask_fg)
    if not resultado_cnn or resultado_cnn.get("classe_id") is None:
        raise RuntimeError("cnn_quebra_clf retornou resultado invalido.")

    probs = resultado_cnn.get("probs") or [0.0, 0.0, 0.0]
    prob_parcial = float(probs[1]) if len(probs) > 1 else 0.0
    prob_quebrado = float(probs[2]) if len(probs) > 2 else 0.0
    score_cnn = float(np.clip(0.5 * prob_parcial + prob_quebrado, 0.0, 1.0))
    classe = ["integro", "parcial", "quebrado"][int(resultado_cnn["classe_id"])]
    metricas = {"cnn_probs": probs}
    return {"score_quebra": score_cnn, "classe": classe, "metricas": metricas, "cnn": resultado_cnn}


def gerar_mascara_fg(img_bgr):
    """Mascara binaria simples: pixels non-zero sao foreground."""
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def carregar_modelo_sam():
    global _SAM_MODEL
    if _SAM_MODEL is not None:
        return _SAM_MODEL
    if not HAS_SAM:
        raise RuntimeError("Instale 'segment-anything' e 'torch' para usar o SAM.")
    if not os.path.isfile(SAM_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint SAM nao encontrado: {SAM_CHECKPOINT}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam_model.to(device=device)
    _SAM_MODEL = sam_model
    return _SAM_MODEL


def carregar_amg_sam():
    global _SAM_AMG
    if _SAM_AMG is not None:
        return _SAM_AMG
    if SamAutomaticMaskGenerator is None:
        raise RuntimeError("SamAutomaticMaskGenerator nao disponivel.")
    sam_model = carregar_modelo_sam()
    _SAM_AMG = SamAutomaticMaskGenerator(sam_model, **SAM_AMG_CFG)
    return _SAM_AMG


def recortar_bbox(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def enquadrar_em_quadrado(img_bgr, alvo=2500, margem=0.9):
    """
    Reenquadra em um quadro quadrado, aplicando auto zoom para ocupar a maior
    parte do alvo (deixa uma margem percentual).
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((alvo, alvo, 3), dtype=img_bgr.dtype)

    # fator que ocupa ate 'margem' do quadro alvo
    fator_max = margem * min(alvo / w, alvo / h)
    escala = fator_max

    novo_w = max(1, int(w * escala))
    novo_h = max(1, int(h * escala))

    interp = cv2.INTER_CUBIC if escala > 1.0 else cv2.INTER_AREA
    redimensionada = cv2.resize(img_bgr, (novo_w, novo_h), interpolation=interp)

    canvas = np.zeros((alvo, alvo, 3), dtype=img_bgr.dtype)
    x0 = max(0, (alvo - novo_w) // 2)
    y0 = max(0, (alvo - novo_h) // 2)
    canvas[y0 : y0 + novo_h, x0 : x0 + novo_w] = redimensionada
    return canvas


def gerar_caminho_unico(base_path):
    if not os.path.exists(base_path):
        return base_path
    nome, ext = os.path.splitext(base_path)
    idx = 1
    while True:
        candidato = f"{nome}_{idx}{ext}"
        if not os.path.exists(candidato):
            return candidato
        idx += 1


def separar_componentes_sam(imagem: ImageFunctions.Imagem, pasta_destino: str, tamanho_alvo=2500, paletas=None):
    """
    Usa o SAM (AutomaticMaskGenerator) para separar componentes e salva cada um
    como PNG reenquadrado em um quadro quadrado de tamanho_alvo.
    """
    if imagem is None or getattr(imagem, "matriz_NumPy", None) is None:
        return []

    try:
        amg = carregar_amg_sam()
    except Exception as exc:  # noqa: BLE001
        mb.showerror("SAM nao disponivel", str(exc))
        return []

    bgr_img = imagem.matriz_NumPy
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    try:
        masks = amg.generate(rgb_img)
    except Exception as exc:  # noqa: BLE001
        mb.showerror("SAM", f"Falha ao gerar mascaras: {exc}")
        return []

    if not masks:
        mb.showinfo("SAM", "Nenhuma mascara gerada.")
        return []

    os.makedirs(pasta_destino, exist_ok=True)
    pasta_excluidas = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camadas_excluidas")
    os.makedirs(pasta_excluidas, exist_ok=True)
    base_nome = os.path.splitext(imagem.nome)[0]
    salvos = []

    def calcular_ratio_exclusao(mask_local, hsv_local):
        """Proporcao de pixels do componente que batem com as faixas de exclusao/fundo."""
        faixas_exc = montar_faixas_por_categoria(paletas, {"exclusao", "fundo"}, tolerancia=0.45)
        faixas_lista = []
        for chave in ("exclusao", "fundo"):
            faixas_lista.extend(faixas_exc.get(chave, []))
        if not faixas_lista:
            return 0.0
        mask_exc = gerar_mascara_por_faixas(hsv_local, faixas_lista)
        if mask_exc is None:
            return 0.0
        # usa somente pixels do componente com V>0 como base (ignora transparencia/fundo preto fora do recorte)
        mask_comp = mask_local.astype(bool)
        mask_validos = np.logical_and(mask_comp, hsv_local[:, :, 2] > 0)
        area_base = float(np.sum(mask_validos))
        if area_base == 0:
            return 0.0
        inter = np.logical_and(mask_validos, mask_exc > 0)
        return float(np.sum(inter)) / area_base

    def validar_mascara_componente(mask_local, rec_local):
        """Aplica a sequencia de filtros locais: area, forma e conteudo HSV."""
        area_total = rec_local.shape[0] * rec_local.shape[1]
        if area_total == 0:
            return False

        # Filtro 1 - area minima de mascara
        area_mask = mask_local.sum()
        if area_mask < 0.05 * area_total:
            return False

        # Filtro 2 - forma compacta via convexidade
        contours, _ = cv2.findContours(
            mask_local.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        convexidade = area / hull_area
        if convexidade < 0.5:
            return False

        # Filtro 3 - conteudo HSV minimo (evita fundo preto)
        hsv_rec = cv2.cvtColor(rec_local, cv2.COLOR_BGR2HSV)
        sat_ok = hsv_rec[:, :, 1] > 60
        val_ok = hsv_rec[:, :, 2] > 60
        conteudo = np.logical_and(sat_ok, val_ok)
        if np.sum(conteudo) < 0.03 * area_total:
            return False
        if np.sum(hsv_rec[:, :, 2] > 30) < 0.10 * area_mask:
            return False

        return True

    def salvar_excluida(rec_local, idx, motivo):
        nome_arq = f"{base_nome}_comp{idx:02d}_{motivo}.png"
        caminho = gerar_caminho_unico(os.path.join(pasta_excluidas, nome_arq))
        cv2.imwrite(caminho, rec_local)
        return caminho

    for idx, mask_data in enumerate(masks, start=1):
        seg = mask_data.get("segmentation")
        if seg is None:
            continue
        mask_bool = seg.astype(bool)
        if not mask_bool.any():
            continue
        bbox = recortar_bbox(mask_bool.astype(np.uint8) * 255)
        if not bbox:
            continue
        x, y, w, h = bbox

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(bgr_img.shape[1], x0 + w)
        y1 = min(bgr_img.shape[0], y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue

        masked = bgr_img.copy()
        masked[~mask_bool] = (0, 0, 0)
        recorte = masked[y0:y1, x0:x1]
        if recorte.size == 0:
            continue

        mask_rec = mask_bool[y0:y1, x0:x1]
        if not validar_mascara_componente(mask_rec, recorte):
            print(f"[separar_componentes_sam] comp {idx}: descartado por filtros basicos")
            salvar_excluida(recorte, idx, "filtros")
            continue

        # filtro 4: proporcao de exclusao; se >10% descarta
        hsv_rec = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
        ratio_exc = calcular_ratio_exclusao(mask_rec, hsv_rec)
        if ratio_exc > 0.10 and paletas:
            print(f"[separar_componentes_sam] comp {idx}: descartado por exclusao {ratio_exc:.3f} (>{0.10})")
            salvar_excluida(recorte, idx, f"exclusao-{ratio_exc:.3f}")
            continue

        enquadrado = enquadrar_em_quadrado(recorte, alvo=tamanho_alvo)

        caminho_saida = os.path.join(pasta_destino, f"{base_nome}_comp{idx:02d}.png")
        caminho_saida = gerar_caminho_unico(caminho_saida)
        print(f"[separar_componentes_sam] salvando {os.path.basename(caminho_saida)} ratio_exc={ratio_exc:.3f}")
        cv2.imwrite(caminho_saida, enquadrado)
        salvos.append(caminho_saida)

    if not salvos:
        mb.showinfo("SAM", "Nenhuma camada valida para salvar.")

    return salvos
