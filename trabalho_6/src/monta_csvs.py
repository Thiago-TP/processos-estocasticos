"""
Este script extrai features para o conjunto de imagens em trabalho_6/imagens/datasets,
escrevendo os resultados com rótulos em arquivos CSV.
As features extraídas podem ser as indicadas pela opção "trabalho 4" ou "trabalho 6", dependendo do argumento fornecido.
Features são estísticas de primeira ordem, como média, desvio padrão, curtose, etc.,
e várias são calculadas a partir dos histogramas das imagens.
CSVs são dividos ainda por quantização do histograma, e os arquivos são nomeados de acordo.
"""

from glob import glob
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from calculo_de_features_trabalho_4 import calcula_features_trabalho_4
from calculo_de_features_trabalho_6 import calcula_features_trabalho_6, rgb_para_cinza
from imageio.v2 import imread


def extrair_features_para_csv(
    dir_imagens: str = "trabalho_6/imagens/datasets",
    dir_saida: str = "trabalho_6/csvs",
    tipo_features: str = "trabalho_6",
    bins: int = 256,
):
    """
    Extrai features para as imagens em dir_imagens e escreve os resultados em CSVs em dir_saida.
    O CSV resultante terá uma coluna "rotulo" indicando o rótulo da imagem (extraído do nome do subdiretório)

    Parâmetros
    ---
    - dir_imagens: diretório contendo as imagens organizadas em subdiretórios por
        rótulo (e.g., "covid", "normal", etc.).
    - dir_saida: diretório onde os arquivos CSV serão escritos.
    - tipo_features: "trabalho_4" ou "trabalho_6", indicando quais features extrair.
    - bins: quantidade de bins a usar para as features que dependem do histograma.

    Retorna
    ---
    - None (escreve arquivos CSV em dir_saida)

    Levanta
    ---
    - ValueError: se tipo_features não for "trabalho_4" nem "trabalho_6".
    """

    if tipo_features == "trabalho_4":
        funcao_calculo_features = calcula_features_trabalho_4
    elif tipo_features == "trabalho_6":
        funcao_calculo_features = calcula_features_trabalho_6
    else:
        raise ValueError(f"Tipo de trabalho desconhecido: {tipo_features}")

    arquivos_imagens = glob(path.join(dir_imagens, "*", "*.*g"))

    dados_csv = []

    for arquivo in arquivos_imagens:
        print(f"    Processando {arquivo}...")
        imagem = rgb_para_cinza(np.asarray(imread(arquivo)).astype(np.uint8))
        features = funcao_calculo_features(imagem, bins=bins)
        rotulo = path.basename(path.dirname(arquivo))
        nome_imagem = path.basename(arquivo)
        dados_csv.append((nome_imagem,) + features + (rotulo,))

    colunas = ["nome_imagem"] + list(features._fields) + ["rotulo"]
    df = pd.DataFrame(dados_csv, columns=colunas)

    # Converte rótulo categórico para numérico
    traducao_rotulos = {
        "alzheimer": 0,
        "covid": 1,
        "brazilian_seeds": 2,
        "brazilian_leaves": 3,
        "skin_cancer": 4,
    }
    df["rotulo"] = df["rotulo"].map(traducao_rotulos)
    df = df.sort_values(by=["rotulo", "nome_imagem"]).reset_index(drop=True)

    # Escreve o DataFrame em CSV
    nome_csv = f"{tipo_features}_features_bins_{bins}.csv"
    df.to_csv(path.join(dir_saida, nome_csv), index=False)
    print(f"    Features extraídas e salvas em {path.join(dir_saida, nome_csv)}")


def plota_matriz_correlacao(csv: str, dir_saida: str = "trabalho_6/plots"):
    nome_csv = path.basename(csv)
    nome_matriz = f"matriz_correlacao_{nome_csv.replace('.csv', '.pdf')}"

    df = pd.read_csv(csv).drop(columns=["nome_imagem", "rotulo"])

    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    figsize = (df.shape[1], df.shape[1])

    plt.figure(figsize=figsize)
    sns.heatmap(
        df.corr(),
        annot=True,
        cmap=cmap,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Matriz de Correlação das Features - " + nome_csv)
    plt.savefig(path.join(dir_saida, nome_matriz), bbox_inches="tight")
    plt.close()


def plota_dispersoes(csv: str, dir_saida: str = "trabalho_6/plots"):
    nome_csv = path.basename(csv)
    nome_plot = f"dispersoes_{nome_csv.replace('.csv', '.pdf')}"

    df = pd.read_csv(csv).drop(columns=["nome_imagem"])

    sns.pairplot(
        df, hue="rotulo", corner=True, markers=["o", "s", "D", "^", "v"], palette="Set2"
    )
    plt.suptitle("Dispersão das Features - " + nome_csv, y=1.02)
    plt.savefig(path.join(dir_saida, nome_plot), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    """
    Extrai features para ambos os tipos de trabalho e todas quantizações 
    e plota matrizes de correlação.
    """

    from time import time

    tipos = ["trabalho_4", "trabalho_6"]
    quantizacoes = [
        8,
        16,
        32,
    ]

    inicio = time()
    for tipo in tipos:
        for bins in quantizacoes:
            # print(f"Extraindo features para {tipo} com {bins} bins...")
            # extrair_features_para_csv(tipo_features=tipo, bins=bins)

            # print(f"Gerando matriz de correlação para {tipo} com {bins} bins...")
            csv_gerado = path.join(
                "trabalho_6/csvs", f"{tipo}_features_bins_{bins}.csv"
            )
            # plota_matriz_correlacao(csv_gerado)

            print(f"Gerando dispersões para {tipo} com {bins} bins...")
            plota_dispersoes(csv_gerado)
    fim = time()

    delta_t = fim - inicio
    print(f"Tempo total: {delta_t:.2f} segundos ({delta_t / 60.0:.2f} minutos)")
