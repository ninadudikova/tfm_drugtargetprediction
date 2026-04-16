# TFM: Automatización de la identificación de nuevas dianas farmacológicas mediante redes PPI
# Elaborado por: Nina Dudikova
# Fecha: mar 2026

#4. MODEL VALIDATION
# Objetivos:
#   - 4.1 Evaluar la estabilidad del modelo mediante validación cruzada estratificada

#0. Importación de paquetes
#Desde la terminal he instalado los paquetes pandas, 
# Una vez instalados los paquetes, los cargo en memoria:
#   pandas: manejo de datos
#   scikit-learn: librería de machine learning
#   matplotlib: generación de gráficos
#   joblib: optimización del flujo de trabajo

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib

# 4.1 Validación cruzada estratificada
# 4.1.1 Carga del dataset y modelo generados previamente
dataset = pd.read_csv ("03_results/dataset.csv")
modelo = joblib.load("03_results/modelo.joblib")

# 4.1.2 Separación de las features del label
# x = columnas de métricas topológicas que usará el modelo para aprender 
# y = columna de labels (0/1)
X = dataset[["degree", "clustering_coefficient", "betweenness_centrality", "closeness_centrality", "pagerank"]]
y = dataset["drug_target"] 

# 4.1.3 Configuración de los parámetros de la validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4.1.4 Cálculo de las métricas por fold
# cross_validate permite calcular varias métricas a la vez
cv_results = cross_validate(modelo, X, y, cv=cv, scoring=["f1", "precision", "recall"])

# 4.1.5 Mostrar los resultados obtenidos
# Guardo el resultado en el archivo de texto y lo muestro en la terminal
with open("03_results/resultados.txt", "a") as f:
    f.write("Resultados de la validación del modelo:\n")
    f.write(f"F1 promedio:            {cv_results['test_f1'].mean():.4f}\n")
    f.write(f"Precision promedio:     {cv_results['test_precision'].mean():.4f}\n")
    f.write(f"Recall promedio:        {cv_results['test_recall'].mean():.4f}\n")
    f.write(f"Desviacion estandar F1: {cv_results['test_f1'].std():.4f}\n")
    f.write("-" * 40 + "\n")
print("Resultados de la validacion")
print(f"F1 promedio:            {cv_results['test_f1'].mean():.4f}")
print(f"Precision promedio:     {cv_results['test_precision'].mean():.4f}")
print(f"Recall promedio:        {cv_results['test_recall'].mean():.4f}")
print(f"Desviacion estandar F1: {cv_results['test_f1'].std():.4f}")