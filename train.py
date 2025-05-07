import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

# ================================
# ✅ Configurar MLflow correctamente
# ================================

# Usar tracking local relativo (funciona en GitHub Actions)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("airline_satisfaction")

# ================================
# Cargar datos preprocesados
# ================================

X_train, y_train = joblib.load("artifacts/train.pkl")
X_val, y_val = joblib.load("artifacts/val.pkl")

# ================================
# Modelos a comparar
# ================================

modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=6, class_weight='balanced', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

resultados = []

# ================================
# Entrenamiento y evaluación
# ================================

with mlflow.start_run():
    for nombre, modelo in modelos.items():
        print(f"\n🚀 Entrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        print(f"✅ {nombre} - Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        print(classification_report(y_val, y_pred, target_names=["No Satisfecho", "Satisfecho"]))

        # Log de parámetros y métricas
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
    # Seleccionar y guardar mejor modelo
    # ================================

    mejor_modelo = max(resultados, key=lambda x: x['f1'])
    print(f"\n🏆 Mejor modelo: {mejor_modelo['nombre']} (F1: {mejor_modelo['f1']:.4f})")

    os.makedirs("models", exist_ok=True)
    modelo_path = "models/mejor_modelo.pkl"
    joblib.dump(mejor_modelo["modelo"], modelo_path)
    print(f"💾 Modelo guardado en {modelo_path}")

    # ================================
    # Log del modelo y artefacto
    # ================================

    mlflow.sklearn.log_model(mejor_modelo["modelo"], artifact_path="modelo_final")
    mlflow.log_artifact(modelo_path, artifact_path="modelo_final")
