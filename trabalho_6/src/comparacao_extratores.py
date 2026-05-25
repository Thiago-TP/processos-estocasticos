"""
Este script compara as features extraídas usando
o código de demonstração de estatísticas de primeira ordem (FIRST_ORDER_STATISTICS_code_demo - Copia.py)
com as features extraídas usando a função calcula_features_trabalho_6 do arquivo calculo_de_features.py.
O objetivo é verificar se ambas as abordagens produzem os mesmos resultados para as mesmas imagens de entrada.
"""

from glob import glob
from os import path

import imageio
import numpy as np
import skimage
from calculo_de_features_original_adaptado_trabalho_6 import (
    Compute_First_Order_Statistics_Features,
)
from calculo_de_features_trabalho_6 import calcula_features_trabalho_6, rgb_para_cinza


def main():
    """Função principal para comparar as features extraídas pelos dois métodos."""

    pasta = path.join("trabalho_6", "imagens", "teste")
    imagens = glob(path.join(pasta, "*.png")) + glob(path.join(pasta, "*.jpg"))
    for arquivo in imagens:
        print(f"Comparando features de {arquivo}...")

        # Extraindo features usando o código de demonstração
        image = skimage.io.imread(arquivo)
        Bins = 16
        features_demo = Compute_First_Order_Statistics_Features(image, Bins)

        # Extraindo features usando a função calcula_features_trabalho_6
        imagem_de_teste = rgb_para_cinza(np.asarray(imageio.v2.imread(arquivo)))
        features_calculo = calcula_features_trabalho_6(imagem_de_teste, bins=16)

        # Comparando as features extraídas pelos dois métodos
        print("Diferenças entre as features extraídas:")
        for feature in features_demo._fields:
            valor_original = getattr(features_demo, feature)
            valor_calculo = getattr(features_calculo, feature)
            diff = abs(valor_original - valor_calculo)
            percentual = (diff / (abs(valor_original) + 1e-10)) * 100
            print(f"    {feature}: {diff:.5f} ({percentual:.3f}%)")
        print()


if __name__ == "__main__":
    main()
