# nodes.py

import os
import folder_paths
import torch
import json
import tempfile

# Message d'erreur si NeMo n'est pas installé
try:
    from nemo.collections.asr.models import ASRModel, EncDecMultiTaskModel
except ImportError:
    print("################################################################")
    print("Canary-ComfyUI: NeMo not found. Please install it by running:")
    print("pip install -U nemo_toolkit['asr']")
    print("in your ComfyUI environment.")
    print("################################################################")

# Dictionnaire pour mettre en cache les modèles chargés et éviter de les recharger
CACHED_MODELS = {}

# --- Fonctions Utilitaires ---

def get_canary_model_list():
    """Retourne la liste des modèles trouvés dans le dossier canary."""
    models_path = folder_paths.get_folder_paths("canary")
    if not models_path:
        return []
    
    models = []
    for model_path in models_path:
        if os.path.exists(model_path):
            models.extend([f for f in os.listdir(model_path) if f.endswith('.nemo')])
    return list(set(models))

def format_timestamps(ts_data):
    """Met en forme les données de timestamp en une chaîne de caractères lisible."""
    if not ts_data:
        return "No timestamp data available."

    output_str = ""
    if 'segment' in ts_data and ts_data['segment']:
        output_str += "--- Segment Timestamps ---\n"
        for stamp in ts_data['segment']:
            output_str += f"{stamp['start']:.2f}s - {stamp['end']:.2f}s : {stamp['segment']}\n"
    
    if 'word' in ts_data and ts_data['word']:
        output_str += "\n--- Word Timestamps ---\n"
        for stamp in ts_data['word']:
            output_str += f"{stamp['start']:.2f}s - {stamp['end']:.2f}s : {stamp['word']}\n"
            
    return output_str.strip() if output_str else "No timestamp data available."


# --- Nodes ---

class CanaryModelLoader:
    """
    Node pour charger un modèle Canary depuis le dossier ComfyUI/models/canary.
    Il détecte automatiquement le type de modèle (ASRModel ou EncDecMultiTaskModel).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_canary_model_list(), ),
            }
        }

    RETURN_TYPES = ("CANARY_MODEL",)
    RETURN_NAMES = ("canary_model",)
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
        
        # Heuristique pour déterminer quelle classe utiliser pour le chargement
        # Basé sur la documentation fournie, 'v2' utilise ASRModel, les autres EncDecMultiTaskModel
        try:
            if 'v2' in model_name:
                model = ASRModel.from_pretrained(model_path)
            else:
                model = EncDecMultiTaskModel.from_pretrained(model_path)
                # Configuration de décodage standard pour les modèles MultiTask
                decode_cfg = model.cfg.decoding
                decode_cfg.beam.beam_size = 1
                model.change_decoding_strategy(decode_cfg)

            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            CACHED_MODELS[model_path] = model
            print(f"Canary-ComfyUI: Model '{model_name}' loaded successfully.")
            return (model,)

        except Exception as e:
            raise RuntimeError(f"Failed to load Canary model {model_name}. Error: {e}")


class CanaryTranscriber:
    """
    Node unique pour la transcription et la traduction.
    Il s'adapte au type de modèle chargé (v2, flash, 1b).
    """
    @classmethod
    def INPUT_TYPES(s):
        # Langues supportées par chaque type de modèle
        LANGS_V2 = ["en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"]
        LANGS_MULTITASK = ["en", "de", "es", "fr"]

        return {
            "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio_path": ("STRING", {"default": "path/to/your/audio.wav"}),
                "task": (["transcribe", "translate"],),
                "source_lang": (LANGS_V2 + list(set(LANGS_MULTITASK) - set(LANGS_V2)), {"default": "en"}),
                "target_lang": (LANGS_V2 + list(set(LANGS_MULTITASK) - set(LANGS_V2)), {"default": "en"}),
                "pnc": (["yes", "no"],),
                "get_timestamps": (["yes", "no"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "timestamps")
    FUNCTION = "transcribe"
    CATEGORY = "Canary-ComfyUI"
    
    def transcribe(self, canary_model, audio_path, task, source_lang, target_lang, pnc, get_timestamps):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")

        device = next(canary_model.parameters()).device
        print(f"Canary-ComfyUI: Running inference on device: {device}")
        
        # --- Logique pour ASRModel (ex: Canary-1b-v2) ---
        if isinstance(canary_model, ASRModel):
            print("Canary-ComfyUI: Using ASRModel (v2) API.")
            if task == "transcribe":
                # Pour la transcription, la langue source et cible doivent être les mêmes
                target_lang = source_lang
            
            output = canary_model.transcribe(
                [audio_path],
                source_lang=source_lang,
                target_lang=target_lang,
                timestamps=(get_timestamps == 'yes')
            )
            
            text_result = output[0].text
            ts_data = output[0].timestamp if hasattr(output[0], 'timestamp') else None
            timestamps_result = format_timestamps(ts_data)
        
        # --- Logique pour EncDecMultiTaskModel (ex: canary-1b, flash) ---
        elif isinstance(canary_model, EncDecMultiTaskModel):
            print("Canary-ComfyUI: Using EncDecMultiTaskModel (flash/1b) API.")
            
            # Création du fichier manifeste temporaire
            manifest = {
                "audio_filepath": audio_path,
                "source_lang": source_lang,
                "target_lang": target_lang if task == "translate" else source_lang,
                "pnc": pnc,
            }

            # Les modèles 'flash' et '1b' ont des manifestes légèrement différents
            model_name = canary_model.cfg.name
            if 'flash' in model_name.lower():
                 manifest["timestamp"] = get_timestamps
            else: # Modèle 'canary-1b' standard
                manifest["duration"] = None # NeMo gère ça automatiquement
                manifest["taskname"] = "s2t_translation" if task == "translate" else "asr"
                manifest["answer"] = "na"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_manifest:
                tmp_manifest.write(json.dumps(manifest) + '\n')
                manifest_path = tmp_manifest.name

            try:
                # Transcrire en utilisant le manifeste
                output = canary_model.transcribe(
                    manifest_path,
                    batch_size=1,
                )
                
                text_result = output[0].text
                ts_data = output[0].timestamp if hasattr(output[0], 'timestamp') else None
                timestamps_result = format_timestamps(ts_data)
                
            finally:
                # S'assurer que le fichier temporaire est supprimé
                os.remove(manifest_path)
                
        else:
            raise TypeError("Unsupported Canary model type.")
            
        return (text_result, timestamps_result)


# --- Mappings pour ComfyUI ---

NODE_CLASS_MAPPINGS = {
    "CanaryModelLoader": CanaryModelLoader,
    "CanaryTranscriber": CanaryTranscriber
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CanaryModelLoader": "Load Canary Model",
    "CanaryTranscriber": "Transcribe with Canary"
}