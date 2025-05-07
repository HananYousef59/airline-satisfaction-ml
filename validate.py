import os
import joblib
import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ================================
# 🔧 Configurar rutas absolutas
# ================================
workspace_dir = os.getcwd()
outputs_dir = os.path.join(workspace_dir, "outputs")
models_dir = os.path.join(workspace_dir, "models")
artifacts_dir = os.path.join(workspace_dir, "artifacts")
mlruns_dir = os.path.join(workspace_dir, "mlruns")

# Crear carpeta de salida si no existe
os.makedirs(outputs_dir, exist_ok=True)

# ================================
# 📦 Cargar modelo y test data
# ================================
modelo_path = os.path.join(models_dir, "mejor_modelo.pkl")
modelo = joblib.load(modelo_path)

X_test, y_test = joblib.load(os.path.join(artifacts_dir, "test.pkl"))

# ================================
# 📌 Configurar MLflow
# ================================
tracking_uri = f"file://{mlruns_dir}"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Evaluación Final - Test Set")

with mlflow.start_run(run_name="Validación con mejor modelo"):
    # ================================
    # 📈 Predicciones y métricas
    # ================================
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("📊 Evaluación final sobre el conjunto de test:")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1-score: {f1:.4f}")
    print("\n🧾 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No Satisfecho", "Satisfecho"]))

    # Log de métricas en MLflow
    mlflow.log_metric("accuracy_test", acc)
    mlflow.log_metric("f1_score_test", f1)

    # ================================
    # 🧩 Matriz de confusión
    # ================================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Satisfecho", "Satisfecho"],
                yticklabels=["No Satisfecho", "Satisfecho"])
    plt.title("Matriz de Confusión - Test Set")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.tight_layout()

    path_img = os.path.join(outputs_dir, "matriz_confusion_test.png")
    plt.savefig(path_img, dpi=300)
    mlflow.log_artifact(path_img)
    plt.show()
