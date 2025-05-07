import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ================================
# 🔧 Configurar rutas seguras
# ================================
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
models_dir = os.path.join(workspace_dir, "models")
artifacts_dir = os.path.join(workspace_dir, "artifacts")
modelo_path = os.path.join(models_dir, "mejor_modelo.pkl")

# Crear directorios si no existen
os.makedirs(mlruns_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# ================================
# 📌 Configurar MLflow
# ================================
tracking_uri = f"file://{mlruns_dir}"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("airline_satisfaction")

# ================================
# 📥 Cargar datos
# ================================
X_train, y_train = joblib.load(os.path.join(artifacts_dir, "train.pkl"))
X_val, y_val = joblib.load(os.path.join(artifacts_dir, "val.pkl"))

# ================================
# 🤖 Modelos a comparar
# ================================
modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=6, class_weight='balanced', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

resultados = []

# ================================
# 🚀 Entrenamiento y validación
# ================================
with mlflow.start_run() as run:
    for nombre, modelo in modelos.items():
        print(f"\n🚀 Entrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        print(f"✅ {nombre} - Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        print(classification_report(y_val, y_pred, target_names=["No Satisfecho", "Satisfecho"]))

        mlflow.log_param(f"model_{nombre}", modelo.__class__.__name__)
        mlflow.log_metric(f"{nombre}_accuracy", acc)
        mlflow.log_metric(f"{nombre}_f1", f1)

        resultados.append({
            "nombre": nombre,
            "modelo": modelo,
            "accuracy": acc,
            "f1": f1
        })

    # ================================
    # 🏆 Selección y guardado del mejor modelo
    # ================================
    mejor_modelo = max(resultados, key=lambda x: x['f1'])
    print(f"\n🏆 Mejor modelo: {mejor_modelo['nombre']} (F1: {mejor_modelo['f1']:.4f})")

    joblib.dump(mejor_modelo["modelo"], modelo_path)
    print(f"💾 Modelo guardado en {modelo_path}")

    # ================================
    # 📦 Log de modelo y artefactos
    # ================================
    mlflow.sklearn.log_model(
        sk_model=mejor_modelo["modelo"],
        artifact_path="modelo_final",
        input_example=X_val[:5],  # ✅ para evitar warnings
    )
    mlflow.log_artifact(modelo_path, artifact_path="modelo_final")
