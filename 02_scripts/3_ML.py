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
#   os: gestión de archivos y rutas del sistema operativo
# Una vez instalados los paquetes, los cargo en memoria:

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# 3.1 Entrenar el modelo Random Forest
# 3.1.1 Carga del dataset
dataset = pd.read_csv("03_results/dataset.csv")

# 3.1.2 Separación de las features del label
# x = columnas de métricas topológicas que usará el modelo para aprender 
# y = columna de labels (0/1)
X = dataset[["degree", "clustering_coefficient"]] 
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
metrics = classification_report(y_test, y_pred)
print(metrics)

# 3.2.3 Generación de la matriz de confusión
# Genera una tabla que muestra cuántos casos el modelo clasificó correcta e incorrectamente
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 3.2.4 Evaluación de la importancia de las features para hacer las predicciones
relevant_features = pd.DataFrame({
    "feature":X.columns,
    "importance": modelo.feature_importances_
}).sort_values("importance", ascending=False)
print(relevant_features)

# 3.3 Guardar el modelo entrenado
import joblib
joblib.dump(modelo,"modelo_trained.joblib")

# Muevo el archivo final al directorio de resultados
os.rename("modelo_trained.joblib", "03_results/modelo.joblib")