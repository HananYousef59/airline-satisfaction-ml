from huggingface_hub import HfApi
import os

# Variables
HF_REPO_ID = "Hanan59/airlineml"  # Tu repo en Hugging Face
MODEL_PATH = "models/mejor_modelo.pkl"  # Ruta de tu modelo local

# Crear API sin token aquí (lo pasamos en la función)
api = HfApi()

# Subir el modelo con autenticación
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="modelo.pkl",
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="Subida automática del modelo desde GitHub Actions",
    token=os.getenv("HF_TOKEN")  # PASA EL TOKEN AQUÍ
)

print("✅ Modelo subido exitosamente a Hugging Face.")
