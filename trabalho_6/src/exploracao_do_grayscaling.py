"""
Este script explora o processo de grayscaling (conversão para escala de cinza) em imagens
através plotagem de imagens no espaço (cubo) RGB antes e depois da conversão.
"""
from glob import glob
from os import path

import matplotlib.pyplot as plt
import numpy as np
from imageio.v2 import imread
from skimage.transform import resize
# from mpl_toolkits.mplot3d import Axes3D

from calculo_de_features import rgb_para_cinza


LIMITE_TAMANHO = 500


def plot_rgb_cubo(axs: list, imagem_rgb: np.ndarray, titulo: str):
    """Plota a imagem RGB em um cubo 3D."""

    # Plota cores no cubo RGB (linha superior)
    for ax, azim in zip(axs[1:], [-22.5, 0, 22.5]):
        ax.scatter(
            xs=imagem_rgb[..., 0], 
            ys=imagem_rgb[..., 1], 
            zs=imagem_rgb[..., 2], 
            c=imagem_rgb.reshape(-1, 3), 
            marker='o',
            alpha=0.1,
        )
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title(titulo)
        ax.view_init(elev=30, azim=azim)
        ax.set_box_aspect([1,1,1])

    # Plota imagem RGB no plano XY (linha inferior)
    axs[0].imshow(imagem_rgb)
    axs[0].set_title(titulo)
    axs[0].axis('off')


def comparar_grayscale(arquivo: str, dir_saida: str = 'trabalho_6/plots'):
    """Função principal para explorar o grayscaling."""

    # Carrega a imagem e normaliza para [0, 1]
    imagem_rgb = np.asarray(imread(arquivo)).astype(np.float64)
    minimos = imagem_rgb.min(axis=(0, 1), keepdims=True)
    maximos = imagem_rgb.max(axis=(0, 1), keepdims=True)
    imagem_rgb = (imagem_rgb - minimos) / (maximos - minimos)

    # Se a figura é muito grande, redimensionar para facilitar a visualização
    if max(imagem_rgb.shape) > LIMITE_TAMANHO:
        fator_redimensionamento = LIMITE_TAMANHO / max(imagem_rgb.shape)
        velha_dim = imagem_rgb.shape
        nova_dimensao = (
            int(velha_dim[0] * fator_redimensionamento), 
            int(velha_dim[1] * fator_redimensionamento)
        )
        imagem_rgb = resize(imagem_rgb, nova_dimensao)
    
    # Plotar a imagem RGB no cubo
    fig = plt.figure(figsize=(24, 12))
    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    plot_rgb_cubo([ax1, ax2, ax3, ax4], imagem_rgb, "Imagem RGB Original")
    
    # Converter a imagem para escala de cinza
    imagem_cinza = rgb_para_cinza(imagem_rgb * maximos.squeeze())  # Multiplica pela escala original para manter a faixa de valores
    
    # Normaliza para [0, 1]
    minimos = imagem_cinza.min(axis=(0, 1), keepdims=True)
    maximos = imagem_cinza.max(axis=(0, 1), keepdims=True)
    imagem_cinza = (imagem_cinza - minimos) / (maximos - minimos)
    
    # Plotar a imagem em escala de cinza no cubo (todos os pontos terão R=G=B)
    ax5 = fig.add_subplot(2, 4, 5)
    ax6 = fig.add_subplot(2, 4, 6, projection='3d')
    ax7 = fig.add_subplot(2, 4, 7, projection='3d')
    ax8 = fig.add_subplot(2, 4, 8, projection='3d')
    plot_rgb_cubo([ax5, ax6, ax7, ax8], np.stack([imagem_cinza]*3, axis=-1), "Imagem em Escala de Cinza")

    plt.tight_layout()
    plt.savefig(f'{dir_saida}/grayscaling_{path.basename(arquivo)}', pad_inches=0.1)
    plt.close()


def main():
    pasta = path.join("trabalho_6", "imagens")
    imagens = glob(path.join(pasta, "*.png")) + glob(path.join(pasta, "*.jpg"))
    for arquivo in imagens:
        print(f"Explorando grayscaling para {arquivo}...")
        comparar_grayscale(arquivo)


if __name__ == "__main__":
    main()