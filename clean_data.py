import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Crear carpeta donde se guardarán los archivos pkl (pero no se suben a GitHub)
os.makedirs("artifacts", exist_ok=True)

# Cargar los datos
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

# Eliminar columnas innecesarias
df_train.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
df_test.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)

# Imputar valores nulos en Arrival Delay con la mediana
median_delay = df_train['Arrival Delay in Minutes'].median()
df_train['Arrival Delay in Minutes'].fillna(median_delay, inplace=True)
df_test['Arrival Delay in Minutes'].fillna(median_delay, inplace=True)

# Convertir variable objetivo a binaria
df_train['satisfaction'] = df_train['satisfaction'].map({
    'satisfied': 1,
    'neutral or dissatisfied': 0
})
df_test['satisfaction'] = df_test['satisfaction'].map({
    'satisfied': 1,
    'neutral or dissatisfied': 0
})

# Codificar variables categóricas con LabelEncoder
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])  # usar mismo encoding
    label_encoders[col] = le  # por si necesitas reutilizar

# Separar X e y
X = df_train.drop("satisfaction", axis=1)
y = df_train["satisfaction"]

X_test = df_test.drop("satisfaction", axis=1)
y_test = df_test["satisfaction"]

# Dividir train en train y val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Guardar datasets como archivos .pkl en la carpeta local 'artifacts'
joblib.dump((X_train, y_train), "artifacts/train.pkl")
joblib.dump((X_val, y_val), "artifacts/val.pkl")
joblib.dump((X_test, y_test), "artifacts/test.pkl")

print("✅ Datos limpios y guardados en artifacts/*.pkl (uso local)")
