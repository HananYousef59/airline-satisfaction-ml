import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow

# Crear carpeta de salida si no existe
os.makedirs("outputs", exist_ok=True)

# Cargar modelo entrenado
modelo = joblib.load("models/mejor_modelo.pkl")

# Cargar datos de prueba
X_test, y_test = joblib.load("artifacts/test.pkl")

# Iniciar experimento de MLflow
mlflow.set_experiment("Evaluación Final - Test Set")

with mlflow.start_run(run_name="Validación con mejor modelo"):
    # Predicciones
    y_pred = modelo.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("📊 Evaluación final sobre el conjunto de test:")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1-score: {f1:.4f}")
    print("\n🧾 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No Satisfecho", "Satisfecho"]))

    # Registrar métricas
    mlflow.log_metric("accuracy_test", acc)
    mlflow.log_metric("f1_score_test", f1)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Visualización
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Satisfecho", "Satisfecho"],
                yticklabels=["No Satisfecho", "Satisfecho"])
    plt.title("Matriz de Confusión - Test Set")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.tight_layout()

    path_img = "outputs/matriz_confusion_test.png"
    plt.savefig(path_img, dpi=300)
    mlflow.log_artifact(path_img)
    plt.show()
