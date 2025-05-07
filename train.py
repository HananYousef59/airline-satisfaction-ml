import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

# Cargar datos preprocesados
X_train, y_train = joblib.load("artifacts/train.pkl")
X_val, y_val = joblib.load("artifacts/val.pkl")

# Modelos a comparar
modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=6, class_weight='balanced', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

resultados = []

# Iniciar experimento MLflow
mlflow.set_tracking_uri("file:./mlruns")  # ✅ Ruta relativa para evitar errores en GitHub Actions
mlflow.set_experiment("airline_satisfaction")

with mlflow.start_run():
    for nombre, modelo in modelos.items():
        print(f"\n🚀 Entrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        print(f"✅ {nombre} - Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        print(classification_report(y_val, y_pred, target_names=["No Satisfecho", "Satisfecho"]))

        # Log de parámetros y métricas en MLflow
        mlflow.log_param(f"model_{nombre}", modelo.__class__.__name__)
        mlflow.log_metric(f"{nombre}_accuracy", acc)
        mlflow.log_metric(f"{nombre}_f1", f1)

        resultados.append({
            "nombre": nombre,
            "modelo": modelo,
            "accuracy": acc,
            "f1": f1
        })

    # Seleccionar mejor modelo
    mejor_modelo = max(resultados, key=lambda x: x['f1'])
    print(f"\n\U0001F3C6 Mejor modelo: {mejor_modelo['nombre']} (F1: {mejor_modelo['f1']:.4f})")

    # Guardar localmente
    os.makedirs("models", exist_ok=True)
    modelo_path = "models/mejor_modelo.pkl"
    joblib.dump(mejor_modelo["modelo"], modelo_path)
    print(f"\U0001F4BE Modelo guardado en {modelo_path}")

    # Log del modelo en MLflow
    mlflow.sklearn.log_model(mejor_modelo["modelo"], "modelo_final")
    mlflow.log_artifact(modelo_path)
