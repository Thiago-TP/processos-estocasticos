"""
Este script realiza a extração de features de uma imagem, retornando um objeto do tipo Features com as estatísticas calculadas.
Imagens devem estar em escala de cinza, i.e., vetores bidimensionais de escalares entre 0 e 255 inclusive.
As features extraídas são as seguintes estatísticas de primeira ordem (vide roteiro do trabalho 4):
- expectância
- moda
- mediana
- amplitude quadrática
- variância
- assimetria (skewness)
- curtose (kurtosis)
- entropia (em bits)
Em particular, a entropia depende da quantização do histograma
"""

from collections import namedtuple

import numpy as np
from scipy.stats import entropy, kurtosis, skew

Features = namedtuple(
    "Features",
    [
        "expectancia",
        "moda",
        "mediana",
        "amplitude_quadratica",
        "variancia",
        "assimetria",
        "curtose",
        "entropia",
    ],
)


def calcula_features_trabalho_4(imagem: np.ndarray, bins: int = 16) -> Features:
    """
    Calcula as features indicadas no roteiro do trabalho 4 para a imagem fornecida.
    A entropia é calculada a partir do histograma da imagem, com a quantidade de bins indicada.
    """

    if imagem.ndim != 2:
        raise ValueError(
            f"A imagem deve estar em escala de cinza (2D), mas tem dimensões {imagem.ndim}!"
        )

    if imagem.dtype != np.uint8:
        raise ValueError(
            f"A imagem deve ter tipo de dado uint8, mas tem tipo {imagem.dtype}!"
        )

    # Flatten da imagem para facilitar os cálculos
    pixels = imagem.flatten().astype(np.float64)

    # Cálculo das features
    expectancia = np.mean(pixels)
    moda = np.bincount(pixels.astype(np.uint8)).argmax()
    mediana = np.median(pixels)
    amplitude_quadratica = (np.max(pixels) - np.min(pixels)) ** 2
    variancia = np.var(pixels)
    assimetria = skew(pixels)
    curtose = kurtosis(pixels)

    # Cálculo do histograma e entropia
    hist, _ = np.histogram(pixels, bins=bins, range=(0, 256))
    hist_normalizado = hist / np.sum(hist)  # Normaliza o histograma para obter uma PMF
    entropia_valor = entropy(hist_normalizado, base=2)  # Entropia em bits

    return Features(
        expectancia=expectancia,
        moda=moda,
        mediana=mediana,
        amplitude_quadratica=amplitude_quadratica,
        variancia=variancia,
        assimetria=assimetria,
        curtose=curtose,
        entropia=entropia_valor,
    )
