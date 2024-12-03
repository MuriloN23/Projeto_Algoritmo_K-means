# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Para visualização em 3D

# Caminhos relativos para os arquivos
caminho_features = "./UCI HAR Dataset/UCI HAR Dataset/features.txt"
caminho_x_train = "./UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt"
caminho_y_train = "./UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt"
caminho_x_test = "./UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt"
caminho_y_test = "./UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt"

# Carregar os dados do dataset
features = pd.read_csv(caminho_features, sep='\s+', header=None)
x_train = pd.read_csv(caminho_x_train, sep='\s+', header=None)
y_train = pd.read_csv(caminho_y_train, sep='\s+', header=None)
x_test = pd.read_csv(caminho_x_test, sep='\s+', header=None)
y_test = pd.read_csv(caminho_y_test, sep='\s+', header=None)

# Combinar os conjuntos de treino e teste
X = pd.concat([x_train, x_test], ignore_index=True)
y = pd.concat([y_train, y_test], ignore_index=True)

# 1. Análise Exploratória de Dados
print("Resumo estatístico:")
print(X.describe())

# Exibir correlações entre as primeiras 10 variáveis
correlacao = X.iloc[:, :10].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação (Primeiras 10 Variáveis)", fontsize=14)
plt.show()

# Escalar os dados
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# 2. Redução de Dimensionalidade com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_escalado)

# 3. Escolha do Número de Clusters (K) com Elbow Method e Silhouette Score
inercia = []
silhouette_scores = []
intervalo_k = range(2, 11)

for k in intervalo_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_pca)
    inercia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Gráfico do Método do Cotovelo
plt.figure(figsize=(10, 6))
plt.plot(intervalo_k, inercia, marker='o')
plt.title('Método do Cotovelo para Determinação de K Ótimo', fontsize=14)
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Inércia (Distância Total Dentro dos Clusters)', fontsize=12)
plt.grid()
plt.show()

# Gráfico do Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(intervalo_k, silhouette_scores, marker='o', color='green')
plt.title('Silhouette Score para Avaliação de K', fontsize=14)
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.grid()
plt.show()

# 4. Implementar K-means com K=6 (baseado nas atividades)
kmeans = KMeans(n_clusters=6, init='k-means++', n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_pca)

# Visualização dos Clusters em 2D
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', s=30, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centróides')
plt.title('Clusters de Atividades Humanas', fontsize=14)
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.legend(['Pontos de Dados (Atividades)', 'Centróides dos Clusters'], loc='best', fontsize=10)
plt.colorbar(scatter, label='Cluster')
plt.grid()
plt.show()

# Métricas Finais
print(f"Silhouette Score Final: {silhouette_score(X_pca, y_kmeans):.2f}")

# Gráfico 3D para Visualização dos Clusters
# Reduzir a dimensionalidade para 3 componentes principais
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_escalado)

# Aplicar K-means com 6 clusters no espaço 3D
kmeans_3d = KMeans(n_clusters=6, init='k-means++', n_init=10, random_state=42)
y_kmeans_3d = kmeans_3d.fit_predict(X_pca_3d)

# Visualizar os clusters em 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y_kmeans_3d, cmap='viridis', s=30, alpha=0.8)
ax.scatter(kmeans_3d.cluster_centers_[:, 0], kmeans_3d.cluster_centers_[:, 1], kmeans_3d.cluster_centers_[:, 2], 
           s=200, c='red', label='Centróides')
ax.set_title('Clusters de Atividades Humanas (Visualização 3D)', fontsize=14)
ax.set_xlabel('Componente Principal 1', fontsize=12)
ax.set_ylabel('Componente Principal 2', fontsize=12)
ax.set_zlabel('Componente Principal 3', fontsize=12)
ax.legend(['Pontos de Dados (Atividades)', 'Centróides dos Clusters'], loc='best', fontsize=10)
fig.colorbar(scatter, label='Cluster')
plt.show()
