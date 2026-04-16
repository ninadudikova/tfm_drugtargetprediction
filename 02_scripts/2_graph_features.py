# TFM: Automatización de la identificación de nuevas dianas farmacológicas mediante redes PPI
# Elaborado por: Nina Dudikova
# Fecha: feb 2026

# 2. GRAPH FEATURES
# Objetivos: 
#   - 2.1 Construir la red PPI a partir de edges.csv
#   - 2.2 Calcular features topológicas por proteína
#   - 2.3 Asignar labels (drug_target = 1 / 0)
#   - 2.4 Generar dataset.csv para ML
#
# Tiempo de ejecución estimado: aproximadamente 45 minutos
#
# 0. Importación de paquetes
# Desde la terminal he instalado los paquetes networkx y pandas
#   networkx: construccion y analisis de PPIN
#   pandas: manejo de datos
#   os: gestión de archivos y rutas del sistema operativo
# Una vez instalados los paquetes, los cargo en memoria:

import pandas as pd
import networkx as nx
import os

# 2.1 Construcción de red PPI

# Cargo los archivos generados previamente 
edge_df = pd.read_csv("03_results/edges.csv")
targets = pd.read_csv("03_results/targets.csv", header=None)[0].dropna().tolist()

# Construyo un grafo no dirigido a partir de la tabla de las interacciones. 
# Cada fila del archivo edge_df se convierte en una conexión entre dos proteínas.

G = nx.from_pandas_edgelist(
    edge_df,
    source="preferredName_A",
    target="preferredName_B",
    edge_attr="score"
)

# 2.2 Cálculo de las features topológicas
# Calculo 2 métricas que son independientes del tamaño del grafo:
#   - grado de nodo: número de proteínas con las que una proteína interactúa directamente.
#   - clustering coefficient: mide si los vecinos de una proteína también están conectados entre sí.
#   - betweenness centrality: mide la importancia de la proteína como puente entre otras proteínas de la red
#   - closeness centrality: mide la proximidad media de la proteína al resto de nodos de la red
#   - pagerank: mide la importancia de la proteína teniendo en cuenta la importancia de sus vecinos

degree = dict(G.degree())
clust_coef = nx.clustering(G)
betweenness = nx.betweenness_centrality(G, normalized=True, weight="score")
closeness = nx.closeness_centrality(G)
pagerank = nx.pagerank(G, weight="score", max_iter=300)

nodes = list(G.nodes()) # Determino los nombre de las proteínas que hay en la red.

# Con las métricas calculadas construyo el nuevo dataset, con una fila por proteína y sus features como columnas
dataset = pd.DataFrame({
    "protein"                : nodes,
    "degree"                 : [degree[n]    for n in nodes],
    "clustering_coefficient" : [clust_coef[n] for n in nodes],
    "betweenness_centrality" : [betweenness[n] for n in nodes],
    "closeness_centrality"   : [closeness[n]   for n in nodes],
    "pagerank"               : [pagerank[n]    for n in nodes],
})

# 2.3 Asignación de labels
# A cada proteína del dataset se le asigna un label:
#   drug_target = 1 para proteínas que son dianas farmacológicas definidas en DrugBank
#   drug_target = 0 para proteínas que aparecen en la red PPI, pero no están en DrugBank

# Convierto la lista de targets a mayúsculas para evitar errores de capitalización.
targets_upper = []
for T in targets:
    targets_upper.append(T.upper())

# Recorro cada proteína del dataset y le asigno la etiqueta correspondiente
labels = []
for p in dataset["protein"]:
    if p.upper() in targets_upper:
        labels.append(1)
    else:
        labels.append(0)
dataset["drug_target"] = labels

# 2.4 Generación del dataset final
# Guardo el dataset en un archivo CSV con una fila por proteína, sus features topológicas y su label (drug_target = 0/1)
dataset.to_csv("dataset.csv", index=False)

# Cuento cuántos positivos y negativos hay en el dataset
n_pos = dataset["drug_target"].sum()
n_neg = len(dataset) - n_pos

# Guardo el resultado en el archivo de texto y lo muestro en la terminal
with open("03_results/resultados.txt", "a") as f:
    f.write("Resultados de construcción del dataset: \n")
    f.write(f"Proteínas en el dataset: {len(dataset)}\n")
    f.write(f"Positivos (drug_target=1): {n_pos}\n")
    f.write(f"Negativos (drug_target=0): {n_neg}\n")
    f.write("-" * 40 + "\n")
print(f"Proteínas en el dataset: {len(dataset)}")
print(f"Positivos (drug_target=1): {n_pos}")
print(f"Negativos (drug_target=0): {n_neg}")

# Muevo el archivo final al directorio de resultados
os.rename("dataset.csv", "03_results/dataset.csv")