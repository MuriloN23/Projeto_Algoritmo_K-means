# **Reconhecimento de Atividades Humanas com K-means**

## **Objetivo do Projeto**

O objetivo deste projeto é aplicar o algoritmo de **K-means** para agrupar atividades humanas com base em dados de sensores de smartphones. O **dataset "Human Activity Recognition Using Smartphones"** contém medições de 561 variáveis extraídas de sinais de acelerômetro e giroscópio de smartphones usados por 30 voluntários realizando atividades diárias, como caminhar, subir escadas, sentar e deitar.

O projeto inclui:
- **Análise Exploratória** dos dados.
- **Pré-processamento**, incluindo normalização e redução de dimensionalidade (PCA).
- **Implementação do K-means** para agrupar as atividades.
- **Avaliação** dos resultados usando o **Método do Cotovelo** e o **Silhouette Score**.
- **Visualização** dos clusters formados usando gráficos 2D e 3D.

## **Instruções para Executar o Código**

### **Requisitos**

1. **Python 3.x** instalado.
2. Bibliotecas necessárias:
   - **pandas**
   - **numpy**
   - **matplotlib**
   - **seaborn**
   - **sklearn**

### **Instalação das Dependências**

Para instalar as bibliotecas necessárias, execute o seguinte comando no seu terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **Baixando o Dataset**

- Você pode acessar o dataset **Human Activity Recognition Using Smartphones** [aqui](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

- Extraia os arquivos para o diretório UCI HAR Dataset dentro do seu projeto.

### **Executando o Código**
- Abra o terminal ou prompt de comando.
- Navegue até o diretório onde o código está armazenado.
- Execute o script Python:

```bash
python Projeto_Algoritmo_K-means.py
```
O código irá:

- Carregar e pré-processar os dados.
- Realizar o agrupamento com K-means.
- Gerar gráficos de análise, incluindo o Método do Cotovelo, Silhouette Score, e visualizações 2D e 3D dos clusters.

### **Estrutura do Projeto**

```bash
.
├── UCI HAR Dataset/           # Dataset extraído da UCI
├── nome_do_arquivo.py         # Script principal com a implementação
├── README.md                  # Este arquivo
└── Resultados/                # Pasta para armazenar gráficos e saídas
```
## **Principais Conclusões e Considerações sobre os Resultados Obtidos**

### **Escolha de K=6**

- Utilizando o Método do Cotovelo, foi possível identificar que K=6 era o número ideal de clusters, o que corresponde ao número de atividades no dataset (como caminhar, subir escadas, etc.).
- O Silhouette Score final foi 0.47, o que indicou que os clusters formados são razoavelmente bem separados e coesos, mas há sobreposição entre algumas atividades, como ficar em pé e sentar.

### **Visualização dos Clusters**

- Gráficos 2D e 3D mostraram a separação entre os clusters de atividades, com algumas sobreposições visíveis entre atividades com padrões de movimento semelhantes.
- A redução de dimensionalidade com PCA foi eficaz para facilitar a visualização dos clusters, mas poderia ter sacrificado informações importantes que afetaram a qualidade da separação dos clusters.

### **Limitações do K-means**

- O K-means assume que os clusters têm forma esférica, o que pode não ser ideal para todos os tipos de dados. Isso impactou a qualidade da separação em casos de clusters com formas mais complexas.
- A inicialização do K-means pode ser sensível a valores iniciais dos centróides. No nosso caso, o uso de K-means++ minimizou esse problema, mas outras abordagens como DBSCAN ou HDBSCAN poderiam ser exploradas para melhorar a separação.

### **Melhorias Futuras**

**1. Explorar Algoritmos de Clustering Não Supervisionados:**


- Testar DBSCAN ou HDBSCAN, que não exigem a definição prévia de K e podem identificar clusters com formas complexas.

**2. Aprimorar a Redução de Dimensionalidade:**

- Explorar técnicas como t-SNE ou UMAP, que preservam melhor as relações locais entre os dados.

**3. Seleção de Variáveis:**

- Avaliar a importância das variáveis no processo de clustering para otimizar a separação entre as atividades.

**4. Ajuste Fino do Número de Clusters:**

- Explorar técnicas adicionais para validar o número ideal de clusters, como validação cruzada ou inércia acumulada.

## **Análise da Matriz de Correlação**

A matriz de correlação foi gerada para as primeiras 10 variáveis do dataset, a fim de entender as relações lineares entre as variáveis e verificar possíveis redundâncias. O cálculo da correlação ajuda a identificar se há variáveis fortemente correlacionadas, o que pode indicar que algumas delas são redundantes e poderiam ser eliminadas para melhorar a eficiência do modelo. Variáveis altamente correlacionadas podem ser combinadas em uma única variável durante a redução de dimensionalidade (por exemplo, utilizando PCA).


- **Correlação entre Variáveis:** Algumas variáveis apresentaram alta correlação positiva, o que é esperado em datasets relacionados a sensores, pois medições de aceleração em eixos semelhantes podem estar altamente correlacionadas. Por exemplo, variáveis que representam aceleração ao longo do eixo X, Y e Z podem ser fortemente correlacionadas.

- **Redundância de Variáveis:** Variáveis altamente correlacionadas podem ser redundantes, o que torna a análise mais complexa sem adicionar valor adicional. A redução de dimensionalidade (como PCA) foi usada para transformar essas variáveis correlacionadas em componentes principais, que são combinações lineares das variáveis originais, preservando a maior parte da variância dos dados.

- **Seleção de Variáveis para Agrupamento:** A análise da matriz de correlação ajudou a confirmar que a redução de dimensionalidade por PCA foi uma escolha acertada, pois ela combina as variáveis mais correlacionadas, simplificando a análise sem perder a informação essencial.

## **Autores:** 

- Murilo Carlos Novais
- Marley Rebouças Ferreira

## **Observações:**
O relatório técnico completo está disponível na pasta 