# TFM: Automatización de la identificación de nuevas dianas farmacológicas mediante redes PPI
# Elaborado por: Nina Dudikova
# Fecha: feb 2026

#3. RANDOM FOREST MACHINE LEARNING
# Objetivos:
#   - 3.1 Entrenar el modelo Random Forest
#   - 3.2 Evaluar el modelo 

#0. Importación de paquetes
#Desde la terminal he instalado los paquetes pandas y scikit-learn
#   pandas: manejo de datos
#   scikit-learn: librería de machine learning
# Una vez instalados los paquetes, los cargo en memoria:

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 3.1 Entrenar el modelo Random Forest
# Carga del dataset
dataset = pd.read_csv("dataset.csv")
