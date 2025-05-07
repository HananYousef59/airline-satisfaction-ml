import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear carpeta para guardar gráficas
os.makedirs("outputs", exist_ok=True)

# Cargar datasets
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# Estructura
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
print("\nColumnas:", df_train.columns.tolist())

# Valores nulos
print("\nValores nulos por columna (train):\n", df_train.isnull().sum())

# Distribución de la variable objetivo
sns.countplot(data=df_train, x='satisfaction')
plt.title("Distribución de satisfacción")
plt.savefig("outputs/distribucion_satisfaccion.png", dpi=300, bbox_inches="tight")
plt.close()

# Variables categóricas vs satisfacción
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for col in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df_train, x=col, hue='satisfaction')
    plt.title(f'{col} vs Satisfaction')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"outputs/{col.replace(' ', '_')}_vs_satisfaction.png", dpi=300, bbox_inches="tight")
    plt.close()

# Correlación entre variables numéricas
numerical = df_train.select_dtypes(include='number')
plt.figure(figsize=(12,10))
sns.heatmap(numerical.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig("outputs/matriz_correlacion.png", dpi=300, bbox_inches="tight")
plt.close()

# Valores únicos en columnas clave
print("\nValores únicos en 'satisfaction':", df_train['satisfaction'].unique())
for col in categorical_columns:
    print(f"Columna '{col}' - valores únicos:", df_train[col].unique())
