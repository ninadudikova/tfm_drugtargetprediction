# TFM: Automatización de la identificación de nuevas dianas farmacológicas mediante redes PPI
# Elaborado por: Nina Dudikova
# Fecha: mar 2026

#3. RANDOM FOREST MACHINE LEARNING
# Objetivos:
#   - 3.1 Entrenar el modelo Random Forest
#   - 3.2 Evaluar el modelo 
#   - 3.3 Guardar el modelo entrenado

#0. Importación de paquetes
#Desde la terminal he instalado los paquetes pandas y scikit-learn
#   pandas: manejo de datos
#   scikit-learn: librería de machine learning
#   matplotlib: generación de gráficos
#   os: gestión de archivos y rutas del sistema operativo
#   joblib: optimización del flujo de trabajo
# Una vez instalados los paquetes, los cargo en memoria:

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import os
import joblib

# 3.1 Entrenar el modelo Random Forest
# 3.1.1 Carga del dataset
dataset = pd.read_csv("03_results/dataset.csv")

# 3.1.2 Separación de las features del label
# x = columnas de métricas topológicas que usará el modelo para aprender 
# y = columna de labels (0/1)

X = dataset[["degree", "clustering_coefficient", "betweenness_centrality", "closeness_centrality", "pagerank"]]
y = dataset["drug_target"] 

# 3.1.3 División en training y test set
# El dataset se divide en dos partes: 80% para entrenar el modelo, 20% para su evaluación. Se fija la semilla aleatoria para garantizar resultados reproducibles.
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

#3.1.4 Creación del modelo
# Creo el modelo RF con los siguientes parámetros:
#   - n_estimators=100 determina que hay 100 árboles de decisión, es el valor por defecto
#   - class_weight=balanced ajusta automáticamente los pesos para compensar el desbalance entre positivos y negativos. Sin este parámetros habría una tendencia a predecir 0 porque hay más valores negativos que positivos
#   - random_state=42 garantiza reproducibilidad 
modelo = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

# 3.1.5 Entrenamiento del modelo con el training set
modelo.fit(X_train, y_train)

# 3.2 Evalucación del modelo
# 3.2.1 Generación de las predicciones usando el test set
y_pred = modelo.predict(X_test)

# 3.2.2 Comparación de las predicciones calculando las métricas:
#   - precision: mide la fiabilidad de las predicciones postivas, es la proporción de proteínas predichas como drug target que realmente lo son
#   - recall: mide la capacidad de del modelo para no perder casos positivos, proporción de drug targets reales que el modelo ha sido capaz de identificar
#   - f1-score: es la media harmónica de precision y recall
# Imprimo el informe de clasificación para evaluar el rendimiento del modelo en el test set.
metrics = classification_report(y_test, y_pred)

# Guardo el resultado en el archivo de texto y lo muestro en la terminal
with open("03_results/resultados.txt", "a") as f:
    f.write("Resultados de entrenamiento del modelo:\n")
    f.write(metrics + "\n")
print(metrics)

# 3.2.3 Generación de la matriz de confusión
# Genera una tabla que muestra cuántos casos el modelo clasificó correcta e incorrectamente
conf_matrix = confusion_matrix(y_test, y_pred)

# Guardo el resultado en el archivo de texto y lo muestro en la terminal
with open("03_results/resultados.txt", "a") as f:
    f.write(str(conf_matrix) + "\n")
print(conf_matrix)

# 3.2.4 Evaluación de la importancia de las features para hacer las predicciones
relevant_features = pd.DataFrame({
    "feature":X.columns,
    "importance": modelo.feature_importances_
}).sort_values("importance", ascending=False)

# Guardo el resultado en el archivo de texto y lo muestro en la terminal
with open("03_results/resultados.txt", "a") as f:
    f.write(relevant_features.to_string() + "\n")
print(relevant_features)

# 3.2.5 Cálculo de la probablidad de cada proteína de ser una diana molecular.
#   [:, 1] selecciona la probabilidad de todas las proteínas de ser drug_target (columna 1).
#   predict_proba devuelve la probabilidad de cada proteína de ser drug target.

y_prob = modelo.predict_proba(X_test)[:, 1]

# 3.2.6 Cálculo de AUC
# AUC (área bajo la curva) representa la probabilidad de que el modelo, si se le da un ejemplo positivo y negativo 
# elegido al azar, clasifique el positivo más alto que el negativo.
auc = roc_auc_score(y_test, y_prob)

# Guardo el resultado en el archivo de texto y lo muestro en la terminal
with open("03_results/resultados.txt", "a") as f:
    f.write("\n")
    f.write(f"AUC: {auc:.4f}\n")
print(f"AUC: {auc:.4f}")

#3.2.7 Genero los valores de la curva ROC
# Curva ROC (característica operativa del receptor) es la representación visual del rendimiento del modelo en todos los umbrales
#   FPR = false positive rate/tasa de falsos posiivos
#   TPR = true positive rate/tasa de verdaderos positivos
#   _ descarta el valor de theresholds que no se va a usar para dibujar la curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)

# 3.2.8 Generación del gráfico de la curva ROC
plt.figure() 
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio") # Dibujo de una línea diagonal discontinua que representa un clasificador aleatorio (AUC = 0.5) y sirve como referencia.
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Random Forest")
plt.legend()
plt.savefig("03_results/curva_roc.png")
plt.show()

# 3.2.9 Curva precision-recall (PR) y el average precision.
#  3.2.9.1 Cálculo de los valores de precision y recall
#   precision: el ratio de clasificaciones correctas de nuestro clasificador (true + false positive)
#   recall: l ratio de positivos detectado en el dataset por nuestro clasificador (true positive + false negative)
#   _ descarta el valor de theresholds que no se va a usar para dibujar la curva
precision, recall, _ = precision_recall_curve(y_test, y_prob)

# 3.2.9.2 Cálculo de average precision.
#   average precision: el área bajo la curva PR.
ap = average_precision_score(y_test, y_prob)
with open("03_results/resultados.txt", "a") as f:
    f.write(f"Average Precision: {ap:.4f}\n")
    f.write("-" * 40 + "\n")
print(f"Average Precision: {ap:.4f}")

# 3.2.9.3 Generación del gráfico de la curva PR
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precisión-Recall - Random Forest")
plt.legend()
plt.savefig("03_results/curva_precision_recall.png")
plt.show()

# 3.3 Guardar el modelo entrenado
joblib.dump(modelo,"modelo_trained.joblib")

# Muevo el archivo final al directorio de resultados
os.rename("modelo_trained.joblib", "03_results/modelo.joblib")