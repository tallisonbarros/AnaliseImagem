
# ------------------- FileFunctions.py
import os
import glob
import shutil

class Pasta:
    def __init__(self, *caminho):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir = os.path.join(base_dir, *caminho)
        self.lista_arquivos = sorted(glob.glob(os.path.join(self.dir, "*")))
        self.quantidade_arquivos = len(self.lista_arquivos)

    def atualizar (self):
        self.lista_arquivos = sorted(glob.glob(os.path.join(self.dir, "*")))

    def filtrar_arquivos(self, extensao):
        extensao = extensao.lower()
        self.lista_arquivos = sorted( [
            arq for arq in self.lista_arquivos
            if os.path.isfile(arq) and arq.lower().endswith(extensao)
        ])

    def exibir_arquivos(self):
        if not self.lista_arquivos:
            print("Nenhum item encontrado.")
            return

        print("\n----------------------------------------")
        print(f"📂 Conteúdo de: {self.dir}")
        print("----------------------------------------")

        for caminho in self.lista_arquivos:
            nome = os.path.basename(caminho)

            if os.path.isdir(caminho):
                print(f"📁 {nome}/")
            else:
                extensao = os.path.splitext(nome)[1].lower().replace('.', '')
                print(f"📄 {nome:<20} ({extensao})")

        print("----------------------------------------\n")

def formatar_input_diretorio(input_texto):
    """
    Converte um texto como:
        "imagens, calib_amarelo"
    em uma lista:
        ["imagens", "calib_amarelo"]

    Remove:
    - aspas
    - espaços extras
    - entradas vazias
    """
    # Remove aspas simples ou duplas
    texto = input_texto.replace('"', '').replace("'", "")

    # Divide por vírgula
    partes = texto.split(",")

    # Remove espaços laterais e ignora strings vazias
    partes_limpas = [p.strip() for p in partes if p.strip()]

    return partes_limpas


def MoverArquivo(dirArquivo, dirDestino):
    if not dirArquivo:
        return { "ok": False, "erro": "Arquivo não informado." }

    try:
        os.makedirs(dirDestino, exist_ok=True)

        nome_arquivo = os.path.basename(dirArquivo)
        destino_final = os.path.join(dirDestino, nome_arquivo)

        shutil.move(dirArquivo, destino_final)

        return { "ok": True, "destino": destino_final }

    except Exception as e:
        return { "ok": False, "erro": str(e) }


def listar_conteudo_formatado(diretorio):
    pasta = Pasta(*formatar_input_diretorio(diretorio))

    # Se não há arquivos → retorna mensagem simples
    if not pasta.lista_arquivos:
        return {
            "ok": False,
            "mensagem": f"Nenhum arquivo encontrado em:\n{pasta.dir}"
        }

    linhas = []
    linhas.append("----------------------------------------")
    linhas.append(f"📂 Conteúdo de: {pasta.dir}")
    linhas.append("----------------------------------------")

    for caminho in pasta.lista_arquivos:
        nome = os.path.basename(caminho)

        if os.path.isdir(caminho):
            linhas.append(f"📁 {nome}/")
        else:
            ext = os.path.splitext(nome)[1].replace('.', '')
            linhas.append(f"📄 {nome:<20} ({ext})")

    linhas.append("----------------------------------------")

    return {
        "ok": True,
        "mensagem": "\n".join(linhas)
    }
