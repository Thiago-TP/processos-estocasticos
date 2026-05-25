"""
Este script é o ponto de entrada para gerar as figuras e tabelas do trabalho 6.
Ele importa as funções necessárias e as executa para criar as visualizações e análises requeridas.
"""

# Instruções no roteiro (adaptado):
# 5.1 – Observe o conjunto de funções que está no arquivo Python chamado
# “calculo_de_features.py”. Dê uma olhada no conjunto de funções.
# Você consegue discernir o objetivo delas em acordo com o que foi estudado em nosso curso.

# 5.2 – Use o conjunto de funções de “calculo_de_features.py” para computar a
# “features” para o conjunto de imagens recebidas.
# Avalie novamente o desempenho para as três técnicas desenvolvidas:
# Naïve Bayes, discriminante quadrático e discriminante linear.

# 5.3 – Observe o banco de imagens de radiografia torácica que está na pasta "datasets/covid".
# Aplique o extrator de features (calculo_de_features.py) para cada conjunto de imagens.
# Separe 80% para treinamento e 20% para teste.
# Avalie o desempenho quanto a capacidade de discriminação de cada conjunto de imagens do banco de dados para as três técnicas implementadas:
# Naïve Bayes, discriminante quadrático e discriminante linear.

from glob import glob
from os import path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main(out_dir: str = "trabalho_6/resultados"):
    # 5.1
    # CSVs são gerados executando o script monta_csvs.py, que extrai as features e escreve os arquivos CSV.
    # Assumindo que isso já foi feito, o item 5.1 está resolvido.

    csvs = glob("trabalho_6/csvs/trabalho_*_features_bins_*.csv")
    for csv in csvs:
        print(f"Avaliando desempenho usando features do arquivo {csv}...")

        # Carrega o CSV
        df = pd.read_csv(csv).drop(columns=["nome_imagem"])

        # debug: dropa classe 0, alzheimer
        df = df[df["rotulo"] != 0]
        # debug: dropa classe 2, brazilian_seeds
        df = df[df["rotulo"] != 2]

        # Remove features altamente correlacionadas (correlação > 0.9) para evitar multicolinearidade
        corr_matrix = df.drop(columns=["rotulo"]).corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > 0.9)
        ]
        if to_drop:
            print(f"    Encontradas {len(to_drop)} features altamente correlacionadas.")
            df = df.drop(columns=to_drop)
            print(f"    Features removidas por alta correlação: {to_drop}")

        # Plota correlação das features restantes
        nome_csv = path.basename(csv)
        nome_matriz = (
            f"matriz_correlacao_condicionada_{nome_csv.replace('.csv', '.pdf')}"
        )

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
        plt.title("Matriz de Correlação das Features Usadas - " + nome_csv)
        plt.savefig(path.join(out_dir, nome_matriz), bbox_inches="tight")
        plt.close()

        # Prepara para o treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=["rotulo"]),
            df["rotulo"],
            test_size=0.2,
            random_state=42,
            stratify=df["rotulo"],
        )

        nb = GaussianNB()
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()

        # Treinamento
        nb.fit(X_train, y_train)
        lda.fit(X_train, y_train)
        qda.fit(X_train, y_train)

        # Teste: acurácia e matriz de confusão
        acuracia_naive_bayes = accuracy_score(y_test, nb.predict(X_test))
        acuracia_lda = accuracy_score(y_test, lda.predict(X_test))
        acuracia_qda = accuracy_score(y_test, qda.predict(X_test))
        matriz_confusao_naive_bayes = confusion_matrix(y_test, nb.predict(X_test))
        matriz_confusao_lda = confusion_matrix(y_test, lda.predict(X_test))
        matriz_confusao_qda = confusion_matrix(y_test, qda.predict(X_test))

        # Salva os resultados em um arquivo numpy comprimido
        nome_csv = path.basename(csv)
        nome_npz = f"resultados_{nome_csv.replace('.csv', '.npz')}"
        np.savez_compressed(
            path.join(out_dir, nome_npz),
            {
                "acuracia_naive_bayes": acuracia_naive_bayes,
                "acuracia_lda": acuracia_lda,
                "acuracia_qda": acuracia_qda,
                "matriz_confusao_naive_bayes": matriz_confusao_naive_bayes,
                "matriz_confusao_lda": matriz_confusao_lda,
                "matriz_confusao_qda": matriz_confusao_qda,
            },
        )


if __name__ == "__main__":
    inicio = time()
    main()
    fim = time()
    print(
        f"Tempo total: {fim - inicio:.2f} segundos ({(fim - inicio) / 60.0:.2f} minutos)"
    )
