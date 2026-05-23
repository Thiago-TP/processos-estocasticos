# -*- coding: utf-8 -*-
"""
Código de extração de features adaptado do original criado pelo prof. dr. Francisco de Assis (Universidade de Brasília).
Entradas são imagens em escala de cinza, i.e., vetores bidimensionais com entradas entre 0 e 255 inclusive.
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
- entropia de Shannon da imagem (em bits);
- entropia de Shannon dos bins do histograma da imagem (em bits);
- energia normalizada da imagem;
- valor RMS (root media square) da imagem;
- desvio absoluto médio das intensidades dos pixels;
- percentil 92.5 das intensidades dos pixels;
- percentil 85 das intensidades dos pixels;
- percentil 15 das intensidades dos pixels;
- percentil 7.5 das intensidades dos pixels;
- intervalo interquartil das intensidades dos pixels;
- uniformidade dos bins do histograma da imagem;
- desvio absoluto médio robusto (RMAD) das intensidades dos pixels,
  calculado considerando apenas os pixels cujas intensidades estão entre os percentis 10 e 90.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from imageio.v2 import imread
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
        "entropia",
        "entropia_dos_bins",
        "energia",
        "rms",
        "desvio_absoluto_medio",
        "p925",
        "p850",
        "p150",
        "p75",
        "amplitude_interquartil",
        "uniformidade",
        "damr",
    ],
)

# Valor pequeno para evitar log(0) na entropia dos bins do histograma.
EPSILON = 2.2e-16


def rgb_para_cinza(imagem_rgb: np.ndarray) -> np.ndarray:
    """Converte uma imagem RGB para escala de cinza usando a fórmula de luminosidade."""
    if imagem_rgb.ndim != 3 or imagem_rgb.shape[2] != 3:
        raise ValueError("A imagem deve ser RGB (3 canais).")
    pesos_rgb = np.array([0.2125, 0.7154, 0.0721])
    return np.dot(imagem_rgb, pesos_rgb).astype(np.uint8)


def calcula_estatisticas_de_primeira_ordem(imagem: np.ndarray, bins: int) -> namedtuple:

    if imagem.ndim != 2:
        raise ValueError("A imagem deve estar em escala de cinza (2D).")

    features = []
    histograma, _ = np.histogram(
        imagem,
        bins=bins,
        range=(0, 256),
        density=False,
    )
    pmf = histograma / sum(histograma)

    minimo = np.min(imagem)
    maximo = np.max(imagem)
    media = np.mean(imagem)
    std = np.std(imagem)
    var = std**2

    mediana = np.median(imagem)
    assimetria = skew(imagem, axis=None)
    curtose = kurtosis(imagem, axis=None)
    amplitude_ao_quadrado = (maximo - minimo) ** 2
    entropia = entropy(pmf, base=2)
    entropia_dos_bins = entropy(histograma, base=2)

    uniformidade = np.sum(pmf**2, axis=None)
    energia = np.sum(imagem**2, axis=None) / imagem.size
    rms = np.sqrt(energia)
    desvio_absoluto_medio = np.sum(np.abs(imagem - media), axis=None) / imagem.size

    p925 = np.percentile(imagem, 92.5)
    p900 = np.percentile(imagem, 90.0)
    p850 = np.percentile(imagem, 85.0)
    p150 = np.percentile(imagem, 15.0)
    p100 = np.percentile(imagem, 10.0)
    p75 = np.percentile(imagem, 7.5)
    amplitude_interquartil = np.percentile(imagem, 75) - np.percentile(imagem, 25)

    mask = (imagem >= p100) & (imagem <= p900)
    damr = np.mean(np.abs(imagem[mask] - mediana))

    features = Features(
        minimo=minimo,
        maximo=maximo,
        media=media,
        std=std,
        var=var,
        mediana=mediana,
        assimetria=assimetria,
        curtose=curtose,
        amplitude_ao_quadrado=amplitude_ao_quadrado,
        entropia=entropia,
        entropia_dos_bins=entropia_dos_bins,
        energia=energia,
        rms=rms,
        desvio_absoluto_medio=desvio_absoluto_medio,
        p925=p925,
        p850=p850,
        p150=p150,
        p75=p75,
        amplitude_interquartil=amplitude_interquartil,
        uniformidade=uniformidade,
        damr=damr,
    )

    return features


if __name__ == "__main__":
    from glob import glob
    from os import path
    from pprint import pprint

    pasta = path.join("trabalho_6", "imagens")
    imagens = glob(path.join(pasta, "*.png")) + glob(path.join(pasta, "*.jpg"))

    for arquivo in imagens:
        print(f"Extraindo features de {arquivo}...")
        imagem_de_teste = rgb_para_cinza(np.asarray(imread(arquivo)))
        features = calcula_estatisticas_de_primeira_ordem(imagem_de_teste, bins=16)
        pprint(dict(features._asdict()))
