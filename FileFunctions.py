# ------------------- FileFunctions.py
import glob
import json
import os
import shutil
import csv
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
_SCORES_LOCK = threading.Lock()


class Pasta:
    def __init__(self, *caminho):
        self.dir = os.path.join(BASE_DIR, *caminho)
        self.lista_arquivos = sorted(glob.glob(os.path.join(self.dir, "*")))
        self.quantidade_arquivos = len(self.lista_arquivos)

    def atualizar(self):
        self.lista_arquivos = sorted(glob.glob(os.path.join(self.dir, "*")))

    def filtrar_arquivos(self, extensao):
        extensao = extensao.lower()
        self.lista_arquivos = sorted(
            [
                arq
                for arq in self.lista_arquivos
                if os.path.isfile(arq) and arq.lower().endswith(extensao)
            ]
        )

    def exibir_arquivos(self):
        if not self.lista_arquivos:
            print("Nenhum item encontrado.")
            return

        print("\n----------------------------------------")
        print(f"Conteudo de: {self.dir}")
        print("----------------------------------------")

        for caminho in self.lista_arquivos:
            nome = os.path.basename(caminho)

            if os.path.isdir(caminho):
                print(f"- {nome}/")
            else:
                extensao = os.path.splitext(nome)[1].lower().replace(".", "")
                print(f"- {nome:<20} ({extensao})")

        print("----------------------------------------\n")


def caminho_config(*partes):
    return os.path.join(CONFIG_DIR, *partes)


def ler_json(caminho, default=None):
    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            try:
                return json.load(arquivo)
            except json.JSONDecodeError:
                arquivo.seek(0)
                linhas = arquivo.readlines()
                itens = []
                for ln in linhas:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        itens.append(json.loads(ln))
                    except Exception:
                        continue
                return itens if itens else default
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def salvar_json(caminho, conteudo):
    try:
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as arquivo:
            json.dump(conteudo, arquivo, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def caminho_scores_json():
    # mantem nome do arquivo para compatibilidade
    return caminho_config("Classificacoes.json")


def caminho_scores_csv():
    # mantem nome do arquivo para compatibilidade
    return caminho_config("Classificacoes.csv")


def registrar_score(registro, json_path=None, csv_path=None):
    """
    Salva o score (nota) + percentuais em JSON e CSV.
    - registro: dict com as chaves desejadas (ex.: arquivo, score, germen, casca, canjica, util, exclusao).
    - json_path/csv_path: caminhos opcionais; se nao informados, usam a pasta config/.
    """
    def _fmt_csv_val(v):
        if isinstance(v, (int, float)):
            return f"{v:.4f}".replace(".", ",")
        return v

    json_path = json_path or caminho_scores_json()
    csv_path = csv_path or caminho_scores_csv()

    registro = dict(registro)
    if "score" not in registro and "nota" in registro:
        registro["score"] = registro.get("nota")
    if "nota" not in registro and "score" in registro:
        registro["nota"] = registro.get("score")
    registro.setdefault("data_hora", datetime.now().isoformat(timespec="seconds"))

    with _SCORES_LOCK:
        # JSON/NDJSON append
        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            # se arquivo existir como lista JSON, migra para NDJSON
            if os.path.exists(json_path):
                try:
                    historico = ler_json(json_path, [])
                    if isinstance(historico, list):
                        with open(json_path, "w", encoding="utf-8") as arq:
                            for item in historico:
                                arq.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            with open(json_path, "a", encoding="utf-8") as arq:
                arq.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # CSV (append com cabecalho quando novo)
        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            escrever_cabecalho = not os.path.exists(csv_path)
            linha_csv = {k: _fmt_csv_val(v) for k, v in registro.items()}
            with open(csv_path, "a", newline="", encoding="utf-8") as arq:
                writer = csv.DictWriter(arq, fieldnames=list(registro.keys()), delimiter=";")
                if escrever_cabecalho:
                    writer.writeheader()
                writer.writerow(linha_csv)
        except Exception:
            pass


# aliases de compatibilidade
def caminho_classificacoes_json():
    return caminho_scores_json()


def caminho_classificacoes_csv():
    return caminho_scores_csv()


def registrar_classificacao(registro, json_path=None, csv_path=None):
    """Alias de compatibilidade; preferir registrar_score."""
    return registrar_score(registro, json_path=json_path, csv_path=csv_path)


def formatar_input_diretorio(input_texto):
    """
    Converte um texto como:
        "imagens, calib_amarelo"
    em uma lista:
        ["imagens", "calib_amarelo"]

    Remove:
    - aspas
    - espacos extras
    - entradas vazias
    """
    texto = input_texto.replace('"', "").replace("'", "")
    partes = texto.split(",")
    partes_limpas = [p.strip() for p in partes if p.strip()]
    return partes_limpas


def MoverArquivo(dirArquivo, dirDestino):
    if not dirArquivo:
        return {"ok": False, "erro": "Arquivo nao informado."}

    try:
        os.makedirs(dirDestino, exist_ok=True)

        nome_arquivo = os.path.basename(dirArquivo)
        destino_final = os.path.join(dirDestino, nome_arquivo)

        shutil.move(dirArquivo, destino_final)

        return {"ok": True, "destino": destino_final}

    except Exception as e:
        return {"ok": False, "erro": str(e)}


def listar_conteudo_formatado(diretorio):
    pasta = Pasta(*formatar_input_diretorio(diretorio))

    if not pasta.lista_arquivos:
        return {
            "ok": False,
            "mensagem": f"Nenhum arquivo encontrado em:\n{pasta.dir}",
        }

    linhas = []
    linhas.append("----------------------------------------")
    linhas.append(f"Conteudo de: {pasta.dir}")
    linhas.append("----------------------------------------")

    for caminho in pasta.lista_arquivos:
        nome = os.path.basename(caminho)

        if os.path.isdir(caminho):
            linhas.append(f"- {nome}/")
        else:
            ext = os.path.splitext(nome)[1].replace(".", "")
            linhas.append(f"- {nome:<20} ({ext})")

    linhas.append("----------------------------------------")

    return {"ok": True, "mensagem": "\n".join(linhas)}


# --- utilitarios especificos do dataset de quebra ---

def ensure_dataset_quebra(root_dir):
    """Garante a existencia de dataset_quebra/0/1/2 dentro da raiz informada e retorna o caminho absoluto."""
    raiz_abs = os.path.abspath(root_dir)
    ds_base = os.path.join(raiz_abs, "dataset_quebra")
    os.makedirs(ds_base, exist_ok=True)
    for cls in ("0", "1", "2"):
        os.makedirs(os.path.join(ds_base, cls), exist_ok=True)
    return ds_base


def caminho_modelo_quebra(root_dir):
    """Caminho padrao do modelo de quebra (.pt) dentro da raiz."""
    return os.path.join(os.path.abspath(root_dir), "cnn_quebra.pt")
