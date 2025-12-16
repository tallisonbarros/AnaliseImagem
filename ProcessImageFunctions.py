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
except Exception:
    HAS_CNN_QUEBRA = False
    cnn_quebra_clf = None

# controle de notificacao unica sobre qual metodo de quebra foi usado
_QUEBRA_NOTIFY = None


def _notificar_mapa_face(modo: str):
    """Exibe informacao uma unica vez sobre o metodo utilizado para quebra."""
    global _QUEBRA_NOTIFY
    if _QUEBRA_NOTIFY == modo:
        return
    _QUEBRA_NOTIFY = modo
    try:
        mb.showinfo("Quebra", f"Usando analise: {modo}")
    except Exception:
        pass

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


def contar_pixel_hsv(Imagem, hsv_min, hsv_max):
    """Conta quantos pixels estao entre min/max usando HSV ja pre-calculado."""
    mascara = cv2.inRange(Imagem.hsv, hsv_min, hsv_max)
    pixels = cv2.countNonZero(mascara)
    return pixels


def gerar_faixas_hsv(paletas, tolerancia=0.10):
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


def processar_valores_hsv(img, faixas):
    resultados = {}
    total = img.total_pixels
    for categoria, lista_faixas in faixas.items():
        soma = 0
        for faixa in lista_faixas:
            soma += contar_pixel_hsv(img, faixa["min"], faixa["max"])
        resultados[categoria] = (soma / total) * 100.0
    return resultados


def analisar_composicao(img, paletas):
    """
    Classifica cada pixel via KNN no espaco HSV usando as cores de referencia das paletas.
    Retorna um dict com percentuais por categoria, considerando a maioria entre os K mais proximos.
    """
    if img is None or paletas is None:
        return {}

    if getattr(img, "matriz_NumPy", None) is None:
        return {}

    hsv_img = getattr(img, "hsv", None)
    if hsv_img is None:
        try:
            hsv_img = cv2.cvtColor(img.matriz_NumPy, cv2.COLOR_BGR2HSV)
            img.hsv = hsv_img
        except Exception:
            return {}

    # categorias esperadas (mesmo sem exemplo) para manter chaves no retorno
    categorias_base = []
    for categoria in paletas.keys():
        if not categoria:
            continue
        cat_key = categoria.lower()
        if cat_key in ("fundo", "exclusao"):
            cat_key = "exclusao"
        if cat_key == "hsvcor":
            continue
        if cat_key not in categorias_base:
            categorias_base.append(cat_key)
    if "exclusao" not in categorias_base:
        categorias_base.append("exclusao")
    base_resultados = {cat: 0.0 for cat in categorias_base}

    # monta base de exemplos (HSV) por categoria
    exemplos = []
    labels = []
    cat_to_idx = {}
    for categoria, dados in paletas.items():
        if not categoria:
            continue
        cat_key = categoria.lower()
        if cat_key in ("fundo", "exclusao"):
            cat_key = "exclusao"
        if cat_key == "hsvcor":
            continue
        cores = list((dados or {}).get("cores", []))
        if not cores and cat_key == "exclusao":
            # garante referencia para fundo preto ou transparencias
            cores = [(0, 0, 0)]
        if not cores:
            continue
        if cat_key not in cat_to_idx:
            cat_to_idx[cat_key] = len(cat_to_idx)
        idx = cat_to_idx[cat_key]
        for (H, S, V) in cores:
            exemplos.append((int(H), int(S), int(V)))
            labels.append(idx)

    # considera transparencia/preto como exclusao direta
    pixels = hsv_img.reshape(-1, 3).astype(np.float32)
    mask_validos = pixels[:, 2] > 0
    total_pixels_img = len(pixels)
    pixels_validos = pixels[mask_validos]
    zeros_v = total_pixels_img - len(pixels_validos)
    if total_pixels_img == 0:
        return base_resultados

    if not exemplos:
        resultados = dict(base_resultados)
        resultados["exclusao"] = (zeros_v / total_pixels_img) * 100.0
        resultados["util"] = 100.0 - resultados["exclusao"]
        for cat in categorias_base:
            if cat != "exclusao":
                resultados.setdefault(cat, 0.0)
        return resultados

    exemplos_arr = np.array(exemplos, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)
    categorias = [None] * len(cat_to_idx)
    for nome, idx in cat_to_idx.items():
        categorias[idx] = nome

    k = min(5, len(exemplos_arr))

    # processa em blocos para evitar estouro de memoria
    bloco = 200000
    contagem = np.zeros(len(categorias), dtype=np.int64)

    ref_h = exemplos_arr[:, 0][None, :]
    ref_s = exemplos_arr[:, 1][None, :]
    ref_v = exemplos_arr[:, 2][None, :]

    for inicio in range(0, len(pixels_validos), bloco):
        fim = min(inicio + bloco, len(pixels_validos))
        chunk = pixels_validos[inicio:fim]

        # distancia circular em H e euclidiana em S/V (formas irregulares no HSV)
        dh = np.abs(chunk[:, [0]] - ref_h)
        dh = np.minimum(dh, 180 - dh) / 180.0
        ds = np.abs(chunk[:, [1]] - ref_s) / 255.0
        dv = np.abs(chunk[:, [2]] - ref_v) / 255.0
        dist = dh * dh + ds * ds + dv * dv

        viz_idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        viz_labels = labels_arr[viz_idx]

        votos = np.zeros((len(chunk), len(categorias)), dtype=np.int16)
        for cid in range(len(categorias)):
            votos[:, cid] = np.sum(viz_labels == cid, axis=1)

        vencedores = np.argmax(votos, axis=1)
        contagem += np.bincount(vencedores, minlength=len(categorias))

    idx_exclusao = cat_to_idx.get("exclusao")
    cont_exclusao_knn = contagem[idx_exclusao] if idx_exclusao is not None else 0
    cont_exclusao_total = cont_exclusao_knn + zeros_v
    total_util_pixels = max(total_pixels_img - cont_exclusao_total, 0)

    resultados = dict(base_resultados)

    # Percentual relativo ao total (exclusao/util)
    resultados["exclusao"] = (cont_exclusao_total / total_pixels_img) * 100.0
    resultados["util"] = 100.0 - resultados["exclusao"]

    # Percentual das categorias validas normalizadas para 100% da parte util
    for i in range(len(categorias)):
        nome_cat = categorias[i]
        if nome_cat == "exclusao":
            continue
        if total_util_pixels == 0:
            resultados[nome_cat] = 0.0
        else:
            resultados[nome_cat] = (contagem[i] / total_util_pixels) * 100.0

    return resultados


def analisar_score(img, paletas, composicao=None, pesos=None):
    """
    Camada 2 - modelo estatistico ponderado para estimar score/nota a partir da composicao.
    Usa regressao linear sobre o historico salvo (Classificacoes.json). Se faltar base, usa pesos default.
    """
    comp = composicao or analisar_composicao(img, paletas)
    if not comp:
        return {"score": 0.0, "metodo": "sem_dados"}

    base = FileFunctions.ler_json(FileFunctions.caminho_scores_json(), []) or []
    X = []
    y = []
    for item in base:
        score = item.get("score", item.get("nota"))
        if score is None:
            continue
        X.append([
            float(item.get("germen", 0.0)),
            float(item.get("casca", 0.0)),
            float(item.get("canjica", 0.0)),
            float(item.get("util", 0.0)),
        ])
        y.append(float(score))

    feat = np.array([
        float(comp.get("germen", 0.0)),
        float(comp.get("casca", 0.0)),
        float(comp.get("canjica", 0.0)),
        float(comp.get("util", 0.0)),
    ], dtype=np.float64)

    # pesos opcionais para ajustar contribuicao (fallback)
    pesos_default = {"germen": 0.5, "casca": 0.2, "canjica": 0.3}
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
            comp.get("canjica", 0.0) * pesos.get("canjica", 0.0)
        )
        media = num / soma_pesos
        pred = (media / 100.0) * 10.0
        metodo = "ponderado"

    return {"score": pred, "metodo": metodo, "composicao": comp}


def analisar_quebra(img: ImageFunctions.Imagem):
    """
    Avalia grau de quebra fisica usando metricas morfologicas/geomtricas.
    Retorna score (0-1), classificacao qualitativa e metricas auxiliares.
    """
    # Calibracao centralizada
    cfg = {
        "min_area_frag_abs": 5.0,       # px minimos para considerar fragmento
        "min_area_frag_rel": 0.005,     # 0.5% da area principal
        "k_frag_density": 0.8,          # intensidade do score exponencial para densidade de fragmentos
        "rug_blur_ksize": 5,            # kernel blur para suavizar contorno (rugosidade multiescala)
        "sobel_ksize": 3,               # kernel sobel para mapas internos
        "blur_grad_ksize": 3,           # blur no gradiente
        "blur_brilho_ksize": 5,         # blur na intensidade
        "peso_prob_brilho": 0.6,        # peso brilho vs gradiente no mapa interno
        "peso_prob_grad": 0.4,
        "pesos_score": {                # combinacao final
            "area": 0.35,
            "rug": 0.1,
            "frag": 0.25,
            "exp": 0.15,
            "int": 0.15,
        },
    }

    if img is None or getattr(img, "matriz_NumPy", None) is None:
        return {"score_quebra": 0.0, "classe": "indefinido", "metricas": {}}

    mask = _gerar_mascara_fg(img.matriz_NumPy)
    if mask is None:
        return {"score_quebra": 0.0, "classe": "indefinido", "metricas": {}}

    # remove ruidos pequenos e identifica componentes
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return {"score_quebra": 0.0, "classe": "indefinido", "metricas": {}}
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(1 + np.argmax(areas))
    mask_fg = np.zeros_like(mask)
    mask_fg[labels == largest_idx] = 255

    # fragmentacao: calcula area de fragmentos fora do principal (apos limiar)
    main_area = float(stats[largest_idx, cv2.CC_STAT_AREA])
    min_area_frag = max(cfg["min_area_frag_abs"], cfg["min_area_frag_rel"] * main_area)
    frag_area = 0.0
    frag_count = 0
    for i in range(1, num_labels):
        if i == largest_idx:
            continue
        a = float(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area_frag:
            frag_area += a
            frag_count += 1
    frag_area_ratio = frag_area / (main_area + frag_area) if (main_area + frag_area) > 0 else 0.0
    frag_density = frag_count / max(np.sqrt(main_area), 1.0)
    k_frag = cfg["k_frag_density"]
    frag_density_score = 1.0 - np.exp(-k_frag * frag_density)
    fragmentacao = float(np.clip(0.6 * frag_area_ratio + 0.4 * frag_density_score, 0.0, 1.0))

    # contornos e hull
    contours, hierarchy = cv2.findContours(mask_fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"score_quebra": 0.0, "classe": "indefinido", "metricas": {}}
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return {"score_quebra": 0.0, "classe": "indefinido", "metricas": {}}
    perim = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    convexidade = area / hull_area if hull_area > 0 else 1.0
    area_deficit_convex = max(0.0, 1.0 - convexidade)

    # area relativa ao esperado (retangulo minimo)
    # forma de referencia: elipse ajustada (melhor que retangulo para graos ovais)
    try:
        ellipse = cv2.fitEllipse(cnt)
        (w, h) = ellipse[1]
    except Exception:
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
    eixo_maior = float(max(w, h))
    area_ellipse = float(np.pi * (max(w, 1e-3) * 0.5) * (max(h, 1e-3) * 0.5))
    area_ref = float(max(area_ellipse, area, 1.0))
    area_ratio = float(area / area_ref)
    area_deficit = max(0.0, 1.0 - min(area_ratio, 1.0))

    # rugosidade via perimetro comparado ao hull (bordas abertas/irregulares)
    hull_perim = float(cv2.arcLength(hull, True))
    rug_base = max(0.0, (perim - hull_perim) / max(hull_perim, 1.0))

    # rugosidade multiescala: perimetro apos suavizar contorno
    mask_blur = cv2.GaussianBlur(mask_fg, (cfg["rug_blur_ksize"], cfg["rug_blur_ksize"]), 0)
    _, mask_blur_bin = cv2.threshold(mask_blur, 0, 255, cv2.THRESH_BINARY)
    cnt_blur_list, _ = cv2.findContours(mask_blur_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnt_blur_list:
        cnt_blur = max(cnt_blur_list, key=cv2.contourArea)
        perim_blur = float(cv2.arcLength(cnt_blur, True))
    else:
        perim_blur = perim
    rug_ms = max(0.0, (perim - perim_blur) / max(perim_blur, 1.0))
    rugosidade = min(max(0.7 * rug_base + 0.3 * rug_ms, 0.0), 1.0)

    # CNN classificador (obrigatorio)
    if not (HAS_CNN_QUEBRA and cnn_quebra_clf is not None):
        return {"score_quebra": 0.0, "classe": "indefinido", "metricas": {}}

    try:
        resultado_cnn = cnn_quebra_clf.classificar_quebra(
            img.matriz_NumPy,
            mask_fg=mask_fg
        )
    except Exception:
        resultado_cnn = None

    if not (resultado_cnn and resultado_cnn.get("classe_id") is not None):
        return {"score_quebra": 0.0, "classe": "indefinido", "metricas": {}}

    probs = resultado_cnn.get("probs") or [0.0, 0.0, 0.0]
    prob_parcial = float(probs[1]) if len(probs) > 1 else 0.0
    prob_quebrado = float(probs[2]) if len(probs) > 2 else 0.0
    score_cnn = float(np.clip(0.5 * prob_parcial + prob_quebrado, 0.0, 1.0))
    classe = ["integro", "parcial", "quebrado"][int(resultado_cnn["classe_id"])]
    metricas = {
        "area": area,
        "area_ref": area_ref,
        "area_ratio": area_ratio,
        "area_deficit": area_deficit,
        "area_deficit_convex": area_deficit_convex,
        "convexidade": convexidade,
        "perimetro": perim,
        "hull_perimetro": hull_perim,
        "rugosidade": rugosidade,
        "fragmentacao": fragmentacao,
        "fragmentos": frag_count,
        "frag_area_ratio": frag_area_ratio,
        "frag_density": frag_density,
        "cnn_probs": probs,
        "eixo_maior": eixo_maior,
    }
    _notificar_mapa_face("CNN classificador (integro/parcial/quebrado)")
    return {"score_quebra": score_cnn, "classe": classe, "metricas": metricas, "cnn": resultado_cnn}


def _gerar_mascara_fg(img_bgr):
    """Mascara binaria simples: pixels non-zero sao foreground."""
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _carregar_sam_base():
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


def _carregar_sam_amg():
    global _SAM_AMG
    if _SAM_AMG is not None:
        return _SAM_AMG
    if SamAutomaticMaskGenerator is None:
        raise RuntimeError("SamAutomaticMaskGenerator nao disponivel.")
    sam_model = _carregar_sam_base()
    _SAM_AMG = SamAutomaticMaskGenerator(sam_model, **SAM_AMG_CFG)
    return _SAM_AMG


def _recortar_bbox(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def _enquadrar_quadrado(img_bgr, alvo=2500, margem=0.9):
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


def _caminho_unico(base_path):
    if not os.path.exists(base_path):
        return base_path
    nome, ext = os.path.splitext(base_path)
    idx = 1
    while True:
        candidato = f"{nome}_{idx}{ext}"
        if not os.path.exists(candidato):
            return candidato
        idx += 1


def separar_componentes(imagem: ImageFunctions.Imagem, pasta_destino: str, tamanho_alvo=2500, paletas=None):
    """
    Usa o SAM (AutomaticMaskGenerator) para separar componentes e salva cada um
    como PNG reenquadrado em um quadro quadrado de tamanho_alvo.
    """
    if imagem is None or getattr(imagem, "matriz_NumPy", None) is None:
        return []

    try:
        amg = _carregar_sam_amg()
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
    base_nome = os.path.splitext(imagem.nome)[0]
    salvos = []

    def _calc_exclusao_ratio(mask_local, hsv_local):
        cores_exc = []
        for nome, dados in (paletas or {}).items():
            key = (nome or "").lower()
            if key in ("exclusao", "fundo"):
                cores_exc.extend((dados or {}).get("cores", []))
        if not cores_exc:
            return 0.0
        mask_exc = np.zeros(mask_local.shape, dtype=np.uint8)
        for (H, S, V) in cores_exc:
            h_delta = int(max(5, H * 0.08))
            s_delta = int(max(10, S * 0.10))
            v_delta = int(max(10, V * 0.10))
            h_min = max(H - h_delta, 0)
            h_max = min(H + h_delta, 179)
            s_min = max(S - s_delta, 0)
            s_max = min(S + s_delta, 255)
            v_min = max(V - v_delta, 0)
            v_max = min(V + v_delta, 255)
            faixa = cv2.inRange(hsv_local, (h_min, s_min, v_min), (h_max, s_max, v_max))
            mask_exc = cv2.bitwise_or(mask_exc, faixa)
        inter = np.logical_and(mask_local, mask_exc > 0)
        return float(np.sum(inter)) / float(max(np.sum(mask_local), 1))

    def _avaliar_mask(mask_local, rec_local):
        area_total = rec_local.shape[0] * rec_local.shape[1]
        if area_total == 0:
            return False
        area_mask = mask_local.sum()
        if area_mask < 0.05 * area_total:
            return False
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
        hsv_rec = cv2.cvtColor(rec_local, cv2.COLOR_BGR2HSV)
        sat_ok = hsv_rec[:, :, 1] > 60
        val_ok = hsv_rec[:, :, 2] > 60
        conteudo = np.logical_and(sat_ok, val_ok)
        if np.sum(conteudo) < 0.03 * area_total:
            return False
        if np.sum(hsv_rec[:, :, 2] > 30) < 0.10 * area_mask:
            return False
        return True

    def _refinar_por_amg(rec_local, ratio_original):
        try:
            novas = amg.generate(cv2.cvtColor(rec_local, cv2.COLOR_BGR2RGB))
        except Exception:
            return None
        melhor = None
        melhor_ratio = None
        for m in novas:
            seg_loc = m.get("segmentation")
            if seg_loc is None:
                continue
            mask_loc = seg_loc.astype(bool)
            if not mask_loc.any():
                continue
            bbox_loc = _recortar_bbox(mask_loc.astype(np.uint8) * 255)
            if not bbox_loc:
                continue
            xl, yl, wl, hl = bbox_loc
            x1l = xl + wl
            y1l = yl + hl
            mask_crop = mask_loc[yl:y1l, xl:x1l]
            rec_crop = rec_local[yl:y1l, xl:x1l]
            if rec_crop.size == 0 or mask_crop.sum() == 0:
                continue
            ratio_exc = _calc_exclusao_ratio(mask_crop, cv2.cvtColor(rec_crop, cv2.COLOR_BGR2HSV))
            if melhor_ratio is None or ratio_exc < melhor_ratio:
                melhor_ratio = ratio_exc
                melhor = (mask_crop, rec_crop)
        if melhor_ratio is None:
            return None
        if melhor_ratio >= ratio_original:
            return None
        return melhor[0], melhor[1], melhor_ratio

    for idx, mask_data in enumerate(masks, start=1):
        seg = mask_data.get("segmentation")
        if seg is None:
            continue
        mask_bool = seg.astype(bool)
        if not mask_bool.any():
            continue
        bbox = _recortar_bbox(mask_bool.astype(np.uint8) * 255)
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
        if not _avaliar_mask(mask_rec, recorte):
            continue

        # filtro 4: proporcao de exclusao; tenta refinar com AMG local se passar de 15%
        hsv_rec = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
        ratio_exc = _calc_exclusao_ratio(mask_rec, hsv_rec)
        print(f"[separar_componentes] comp {idx}: exclusao_ratio={ratio_exc:.3f}")
        if ratio_exc > 0.15 and paletas:
            refinado = _refinar_por_amg(recorte, ratio_exc)
            if refinado is None:
                print(f"[separar_componentes] comp {idx}: refinado sem melhora, descartado")
                continue
            mask_rec, recorte, ratio_ref = refinado
            print(f"[separar_componentes] comp {idx}: refinado_ratio={ratio_ref:.3f}")
            if not _avaliar_mask(mask_rec, recorte):
                continue

        enquadrado = _enquadrar_quadrado(recorte, alvo=tamanho_alvo)

        caminho_saida = os.path.join(pasta_destino, f"{base_nome}_comp{idx:02d}.png")
        caminho_saida = _caminho_unico(caminho_saida)
        cv2.imwrite(caminho_saida, enquadrado)
        salvos.append(caminho_saida)

    if not salvos:
        mb.showinfo("SAM", "Nenhuma camada valida para salvar.")

    return salvos
