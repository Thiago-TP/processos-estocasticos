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


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)  # Para reprodutibilidade dos resultados

LIMITE_CORRELACAO = 0.95  # Limite para considerar duas features como altamente correlacionadas


def plota_correlacao_da_classe(csv: str, X_train_classe: pd.DataFrame, classe: str, out_dir: str):

    nome_csv = path.basename(csv)
    nome_matriz = (
        f"matriz_correlacao_condicionada_classe_{classe}_{nome_csv.replace('.csv', '.pdf')}"
    )

    mask = np.triu(np.ones_like(X_train_classe.corr(), dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    figsize = (X_train_classe.shape[1], X_train_classe.shape[1])

    plt.figure(figsize=figsize)
    sns.heatmap(
        X_train_classe.corr(),
        annot=True,
        cmap=cmap,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.title(f"Matriz de Correlação da Classe {classe} - {nome_csv}")
    plt.savefig(path.join(out_dir, nome_matriz), bbox_inches="tight")
    plt.close()

    print(f"    Classe {classe}: Matriz de correlação salva como {nome_matriz}.")


def plota_matriz_de_confusao(csv: str, matriz_confusao: np.ndarray, classificador: str, out_dir: str):
    nome_csv = path.basename(csv)
    nome_matriz = f"matriz_confusao_{classificador}_{nome_csv.replace('.csv', '.pdf')}"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matriz_confusao,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
    )
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title(f"Matriz de Confusão - {classificador} - {nome_csv}")
    plt.savefig(path.join(out_dir, nome_matriz), bbox_inches="tight")
    plt.close()


def main(out_dir: str = "trabalho_6/resultados"):
    # 5.1
    # CSVs são gerados executando o script monta_csvs.py, que extrai as features e escreve os arquivos CSV.
    # Assumindo que isso já foi feito, o item 5.1 está resolvido.

    csvs = glob("trabalho_6/csvs/trabalho_*_features_bins_*-PARCIAL.csv")
    for csv in csvs:
        print(f"Avaliando desempenho usando features do arquivo {csv}...")

        # Carrega o CSV
        df = pd.read_csv(csv, index_col="nome_imagem")


        # Separa dados em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=["rotulo"]),
            df["rotulo"],
            test_size=0.2,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

        # Remove do treino, teste features (colunas) que deem correlação > 0.9
        features_constantes = set()
        features_correlacionadas = set()
        for classe in y_train.sort_values().unique():
            X_train_classe = X_train[y_train == classe]
            for coluna in X_train_classe.columns:
                if X_train_classe[coluna].nunique() <= 1:
                    features_constantes.add(coluna)
            corr_matrix = X_train_classe.drop(columns=features_constantes).corr(numeric_only=True)
            print(f"    Correlação para classe {classe}:")
            print(corr_matrix.to_string())
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            features_classe = [
                column for column in upper_tri.columns if any(upper_tri[column].abs() >= LIMITE_CORRELACAO)
            ]
            if features_classe:
                print(f"    Encontradas {len(features_classe)} features altamente correlacionadas.")
                features_correlacionadas.update(features_classe)
                print(f"    Features removidas por alta correlação: {features_classe}")

        X_train = X_train.drop(columns=features_correlacionadas.union(features_constantes))
        X_test = X_test.drop(columns=features_correlacionadas.union(features_constantes))

        plota_correlacao_da_classe(csv, X_train, "todas", out_dir)

        # Instancia os classificadores
        nb = GaussianNB()
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()

        # Treinamento
        nb.fit(X_train, y_train)
        lda.fit(X_train, y_train)
        qda.fit(X_train, y_train)

        # Resultados no teste: acurácia
        acuracia_naive_bayes = accuracy_score(y_test, nb.predict(X_test))
        acuracia_lda = accuracy_score(y_test, lda.predict(X_test))
        acuracia_qda = accuracy_score(y_test, qda.predict(X_test))

        # Resultados no teste: matriz de confusão
        matriz_confusao_naive_bayes = confusion_matrix(y_test, nb.predict(X_test))
        matriz_confusao_lda = confusion_matrix(y_test, lda.predict(X_test))
        matriz_confusao_qda = confusion_matrix(y_test, qda.predict(X_test))

        plota_matriz_de_confusao(csv, matriz_confusao_naive_bayes, "Naive_Bayes", out_dir)
        plota_matriz_de_confusao(csv, matriz_confusao_lda, "LDA", out_dir)
        plota_matriz_de_confusao(csv, matriz_confusao_qda, "QDA", out_dir)

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
