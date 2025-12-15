# ------------------- FileFunctions.py
import glob
import json
import os
import shutil
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")


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
            return json.load(arquivo)
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

    # garante timestamp padrao
    registro = dict(registro)
    # compatibilidade: aceita 'nota' e garante 'score'
    if "score" not in registro and "nota" in registro:
        registro["score"] = registro.get("nota")
    if "nota" not in registro and "score" in registro:
        registro["nota"] = registro.get("score")
    registro.setdefault("data_hora", datetime.now().isoformat(timespec="seconds"))

    # JSON (lista acumulada)
    historico = ler_json(json_path, [])
    if not isinstance(historico, list):
        historico = []
    historico.append(registro)
    salvar_json(json_path, historico)

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
