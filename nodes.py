# nodes.py

import os
import torch
import folder_paths

# Import de NeMo avec gestion d'erreur (une seule fois ici)
try:
    from nemo.collections.asr.models import ASRModel
except ImportError:
    print("################################################################")
    print("Canary-ComfyUI: NeMo not found. Please install it by running:")
    print("pip install -U nemo_toolkit['asr'] scipy numpy torchaudio")
    print("in your ComfyUI environment.")
    print("################################################################")
    # On arrête le chargement si NeMo n'est pas là pour éviter plus d'erreurs
    raise ImportError("NeMo toolkit is required for Canary-ComfyUI nodes.")

# --- Configuration et Utilitaires Communs ---

# Enregistrement du dossier de modèles
canary_models_path = os.path.join(folder_paths.models_dir, "canary")
if not os.path.exists(canary_models_path):
    try:
        os.makedirs(canary_models_path)
        print(f"Canary-ComfyUI: Created directory {canary_models_path}")
    except OSError as e:
        print(f"Canary-ComfyUI: Error creating directory {canary_models_path}: {e}")
folder_paths.add_model_folder_path("canary", canary_models_path)

# Cache pour les modèles chargés
CACHED_MODELS = {}

def get_canary_model_list():
    """Retourne la liste des fichiers de modèles Canary disponibles."""
    return folder_paths.get_filename_list("canary")

# --- Nœuds Communs ---

class CanaryModelLoader:
    """
    Nœud générique pour charger n'importe quel modèle ASR de type NeMo.
    """
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "model_name": (get_canary_model_list(), ), } }
    
    RETURN_TYPES = ("CANARY_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Canary-ComfyUI"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("canary", model_name)
        if not model_path:
            raise FileNotFoundError(f"Model {model_name} not found.")
            
        if model_path in CACHED_MODELS:
            print(f"Canary-ComfyUI: Loading cached model '{model_name}'")
            return (CACHED_MODELS[model_path],)
        
        print(f"Canary-ComfyUI: Loading model '{model_name}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = ASRModel.restore_from(restore_path=model_path, map_location=device)
            model.to(device)
            model.eval()
            CACHED_MODELS[model_path] = model
            print(f"Canary-ComfyUI: Model '{model_name}' loaded successfully on {device}.")
            return (model,)
        except Exception as e:
            print(f"Canary-ComfyUI: An error occurred while loading the model: {e}")
            raise e

# --- Mappings pour les nœuds de CE fichier ---
NODE_CLASS_MAPPINGS = {
    "CanaryModelLoader": CanaryModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CanaryModelLoader": "Load Canary Model",
}