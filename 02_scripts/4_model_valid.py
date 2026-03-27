# TFM: Automatización de la identificación de nuevas dianas farmacológicas mediante redes PPI
# Elaborado por: Nina Dudikova
# Fecha: mar 2026

#4. MODEL VALIDATION
# Objetivos:
#   - 4.1 Evaluar la estabilidad del modelo mediante validación cruzada estratificada
#   - 4.2 Evaluar la capacidad discriminativa del modelo mediante la curva ROC y el AUC
#   - 4.3 Evaluar el rendimiento del modelo en la identificación de drug targets mediante la curva precisión-recall y el Average Precision

#0. Importación de paquetes
#Desde la terminal he instalado los paquetes pandas, 
# Una vez instalados los paquetes, los cargo en memoria:
#   pandas: manejo de datos
#   scikit-learn: librería de machine learning
#   matplotlib: generación de gráficos
#   joblib: optimización del flujo de trabajo

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import joblib

# 4.1 Validación cruzada estratificada
# 4.1.1 Carga del dataset y modelo generados previamente
dataset = pd.read_csv ("03_results/dataset.csv")
modelo = joblib.load("03_results/modelo.joblib")

# 4.1.2 Separación de las features del label
# x = columnas de métricas topológicas que usará el modelo para aprender 
# y = columna de labels (0/1)
X = dataset[["degree", "clustering_coefficient"]] 
y = dataset["drug_target"] 

# 4.1.3 Configuración de los parámetros de la validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4.1.4 Cálculo de los scores
scores = cross_val_score(modelo, X, y, cv=cv, scoring='f1')

# 4.1.5 Mostrar los resultados obtenidos
print ("Resultados de la validación")
print(f"Puntuaciones F1 por fold: {scores}")
print(f"F1-Score promedio: {scores.mean():.4f}")
print(f"Desviación estándar: {scores.std():.4f}")

# 4.2 Curva ROC y el AUC.
# El dataset se divide en dos partes: 80% para entrenar el modelo, 20% para su evaluación. Se fija la semilla aleatoria para garantizar resultados reproducibles.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# 4.2.2 Cálculo de la probablidad de cada proteína de ser una diana molecular.
#   - [:, 1] selecciona la probabilidad de todas las proteínas de ser drug_target (columna 1).
y_prob = modelo.predict_proba(X_test)[:, 1]

# 4.2.3 Cálculo de AUC
# AUC (área bajo la curva) representa la probabilidad de que el modelo, si se le da un ejemplo positivo y negativo 
# elegido al azar, clasifique el positivo más alto que el negativo.
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.4f}")

# 4.2.4 Genero los valores de la curva ROC
# Curva ROC (característica operativa del receptor) es la representación visual del rendimiento del modelo en todos los umbrales
#   FPR = false positive rate/tasa de falsos posiivos
#   TPR = true positive rate/tasa de verdaderos positivos
#   _ descarta el valor de theresholds que no se va a usar para dibujar la curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)

# 4.2.5 Generación del gráfico de la curva ROC
plt.figure() 
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio") # Dibujo de una línea diagonal discontinua que representa un clasificador aleatorio (AUC = 0.5) y sirve como referencia.
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Random Forest")
plt.legend()
plt.savefig("03_results/curva_roc.png")
plt.show()

# 4.3 Curva precision-recall (PR) y el average precision.
#  4.3.1 Cálculo de los valores de precision y recall
#   precision: el ratio de clasificaciones correctas de nuestro clasificador (true + false positive)
#   recall: l ratio de positivos detectado en el dataset por nuestro clasificador (true positive + false negative)
#   _ descarta el valor de theresholds que no se va a usar para dibujar la curva
precision, recall, _ = precision_recall_curve(y_test, y_prob)

# 4.3.2 Cálculo de average precision.
#   average precision: el área bajo la curva PR.
ap = average_precision_score(y_test, y_prob)
print(f"Average Precision: {ap:.4f}")

# 4.3.3 Generación del gráfico de la curva PR
plt.figure()
plt.plot(recall, precision, label=f"AP = {ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precisión-Recall - Random Forest")
plt.legend()
plt.savefig("03_results/curva_precision_recall.png")
plt.show()