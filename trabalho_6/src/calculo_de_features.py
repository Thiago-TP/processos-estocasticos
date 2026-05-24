# -*- coding: utf-8 -*-
"""
Código de extração de features adaptado do original criado pelo prof. dr. Francisco de Assis (Universidade de Brasília).
Entradas são imagens em escala de cinza, i.e., vetores bidimensionais de escalares entre 0 e 255 inclusive.
Features extraídas são as seguintes estatísticas de primeira ordem:
- mínima intensidade do pixel;
- máxima intensidade do pixel;
- média das intensidades dos pixels;
- desvio padrão das intensidades dos pixels;
- variância das intensidades dos pixels;
- mediana das intensidades dos pixels;
- assimetria (skewness) das intensidades dos pixels;
- curtose (kurtosis) das intensidades dos pixels;
- quadrado da diferença entre a máxima e a mínima intensidade dos pixels;
- entropia de Shannon da imagem com 256 bins (em bits);
- entropia de Shannon da imagem com uma dada quantidade de bins (em bits);
- energia media da imagem;
- valor RMS (root mean square) da imagem;
- desvio absoluto médio das intensidades dos pixels;
- percentil 92.5 das intensidades dos pixels;
- percentil 85 das intensidades dos pixels;
- percentil 15 das intensidades dos pixels;
- percentil 7.5 das intensidades dos pixels;
- amplitude interquartil das intensidades dos pixels;
- uniformidade dos bins do histograma da imagem (equivalente à energia da PMF dos bins);
- desvio absoluto médio robusto das intensidades dos pixels,
  calculado considerando apenas os pixels cujas intensidades estão entre os percentis 10 e 90.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.stats import entropy, kurtosis, skew

Features = namedtuple(
    "Features",
    [
        "minimo",
        "maximo",
        "media",
        "std",
        "var",
        "mediana",
        "assimetria",
        "curtose",
        "amplitude_ao_quadrado",
        "entropia_completa",
        "entropia_dos_bins",
        "energia_media",
        "rms",
        "desvio_absoluto_medio",
        "p925",
        "p850",
        "p150",
        "p75",
        "amplitude_interquartil",
        "uniformidade",
        "media_robusta",
    ],
)

# Valor pequeno para evitar log(0) na entropia dos bins do histograma.
EPSILON = 2.2e-16


def rgb_para_cinza(imagem_rgb: np.ndarray) -> np.ndarray:
    """
    Converte uma imagem RGB para escala de cinza usando a fórmula de luminosidade.
    
    Parâmetros
    ---
    - imagem_rgb: np.ndarray
        Imagem RGB de entrada, com formato (altura, largura, 3), 
        sendo os canais na ordem R, G, B e valores entre 0 e 255 inclusive.
    
    Retorna
    ---
    - np.ndarray
        Imagem em escala de cinza, com formato (altura, largura) 
        e valores inteiros sem sinal entre 0 e 255 inclusive.

    Levanta
    ---
    - ValueError: Se a imagem de entrada não for RGB (3 canais).
    """
    if imagem_rgb.ndim != 3 or imagem_rgb.shape[2] != 3:
        raise ValueError("A imagem deve ser RGB (3 canais).")
    pesos_rgb = np.array([0.2125, 0.7154, 0.0721])
    return np.dot(imagem_rgb, pesos_rgb).astype(np.uint8)


def calcula_estatisticas_de_primeira_ordem(imagem: np.ndarray, bins: int) -> namedtuple:

    if imagem.ndim != 2:
        raise ValueError(
            f"A imagem deve estar em escala de cinza (2D), mas tem dimensões {imagem.ndim}!"
        )
    
    if imagem.dtype != np.uint8:
        raise ValueError(
            f"A imagem deve ter tipo de dado uint8, mas tem tipo {imagem.dtype}!"
        )
    
    # Achata, converte imagem em float para os cálculos
    imagem = imagem.flatten().astype(np.float64)

    # Histograma do opencv estava dando muito problema
    histograma, _ = np.histogram(
        imagem,
        bins=bins,
        range=(0, 256),
        density=False,
    )
    pmf = histograma / sum(histograma)

    # Grandezas estatísticas diretas da imagem
    minimo = np.min(imagem)
    maximo = np.max(imagem)
    media = np.mean(imagem)
    std = np.std(imagem)
    var = std**2
    mediana = np.median(imagem)
    assimetria = skew(imagem, axis=None)
    curtose = kurtosis(imagem, axis=None)
    amplitude_ao_quadrado = (maximo - minimo) ** 2
    desvio_absoluto_medio = np.sum(np.abs(imagem - media), axis=None) / imagem.size

    # Entropia de Shannon da imagem e dos bins do histograma
    # Obs.: entropia dos bins é uma aproximação da entropiada imagem,
    # de forma que ambas são equivalentes quando bins=256
    _, contagens = np.unique(imagem, return_counts=True)
    entropia_completa = entropy(contagens, base=2)
    entropia_dos_bins = entropy(pmf, base=2)

    # Energia por pixel e RMS da imagem
    energia_media = np.sum(imagem**2, axis=None) / imagem.size
    rms = np.sqrt(energia_media)

    # Energia da PMF (limitada pelos bins)
    uniformidade = np.sum(pmf**2, axis=None)

    # Percentis e intervalo interquartil
    p925 = np.percentile(imagem, 92.5)
    p900 = np.percentile(imagem, 90.0)
    p850 = np.percentile(imagem, 85.0)
    p150 = np.percentile(imagem, 15.0)
    p100 = np.percentile(imagem, 10.0)
    p75 = np.percentile(imagem, 7.5)
    amplitude_interquartil = np.percentile(imagem, 75) - np.percentile(imagem, 25)

    # A implementação original do rMAD / damr era equivalente 
    # à média dos pixels de percentis 10 entre e 90.
    # Não creio que isso configure um desvio, mas segue uma reimplementação 
    # mais legível e eficiente do mesmo cálculo.
    mask = (imagem >= p100) & (imagem <= p900)
    media_robusta = np.mean(imagem[mask])

    return Features(
        minimo=minimo,
        maximo=maximo,
        media=media,
        std=std,
        var=var,
        mediana=mediana,
        assimetria=assimetria,
        curtose=curtose,
        amplitude_ao_quadrado=amplitude_ao_quadrado,
        entropia_completa=entropia_completa,
        entropia_dos_bins=entropia_dos_bins,
        energia_media=energia_media,
        rms=rms,
        desvio_absoluto_medio=desvio_absoluto_medio,
        p925=p925,
        p850=p850,
        p150=p150,
        p75=p75,
        amplitude_interquartil=amplitude_interquartil,
        uniformidade=uniformidade,
        media_robusta=media_robusta,
    )


if __name__ == "__main__":
    """Sugestão de uso das funções neste script."""
    from glob import glob
    from os import path
    from pprint import pprint

    from imageio.v2 import imread

    pasta = path.join("trabalho_6", "imagens")
    imagens = glob(path.join(pasta, "*.png")) + glob(path.join(pasta, "*.jpg"))

    for arquivo in imagens:
        print(f"Extraindo features de {arquivo}...")
        imagem_de_teste = rgb_para_cinza(np.asarray(imread(arquivo)))
        features = calcula_estatisticas_de_primeira_ordem(imagem_de_teste, bins=16)
        pprint(dict(features._asdict()))
