# nodes.py

import os
import folder_paths
import torch
import tempfile
import numpy as np
import torchaudio
from scipy.io.wavfile import write as write_wav

# Enregistrement du dossier de modèles
canary_models_path = os.path.join(folder_paths.models_dir, "canary")
if not os.path.exists(canary_models_path):
    try:
        os.makedirs(canary_models_path)
        print(f"Canary-ComfyUI: Created directory {canary_models_path}")
    except OSError as e:
        print(f"Canary-ComfyUI: Error creating directory {canary_models_path}: {e}")
folder_paths.add_model_folder_path("canary", canary_models_path)

# Import de NeMo avec gestion d'erreur
try:
    from nemo.collections.asr.models import ASRModel
except ImportError:
    print("################################################################")
    print("Canary-ComfyUI: NeMo not found. Please install it by running:")
    print("pip install -U nemo_toolkit['asr'] scipy numpy torchaudio")
    print("in your ComfyUI environment.")
    print("################################################################")

# --- Constantes et Cache ---

CACHED_MODELS = {}
ASR_LANGS = sorted(["en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"])
TRANSLATION_LANGS = sorted([lang for lang in ASR_LANGS if lang != 'en'])

# --- Fonctions Utilitaires ---

def get_canary_model_list():
    return folder_paths.get_filename_list("canary")

def process_audio_and_run_model(model, audio, source_lang, target_lang):
    """
    Fonction helper simplifiée : prépare l'audio, exécute le modèle Canary 
    et retourne UNIQUEMENT le texte résultant.
    """
    waveform, sample_rate = audio["waveform"], audio["sample_rate"]
    
    if waveform.ndim > 2: waveform = waveform[0]
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze()

    if sample_rate != 16000:
        print(f"Canary-ComfyUI: Resampling audio from {sample_rate} Hz to 16000 Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    audio_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            audio_path = temp_audio_file.name
            np_waveform = waveform.cpu().numpy()
            scaled_waveform = np.int16(np_waveform / np.abs(np_waveform).max() * 32767) if np.abs(np_waveform).max() > 0 else np.int16(np_waveform)
            write_wav(audio_path, 16000, scaled_waveform)

        device = next(model.parameters()).device
        task_name = "Translation" if source_lang != target_lang else "Transcription"
        print(f"Canary-ComfyUI: Running {task_name} (source: {source_lang}, target: {target_lang}) on device: {device}")
        
        # Appel au modèle SANS l'option timestamps pour garantir la stabilité
        output = model.transcribe(
            [audio_path],
            source_lang=source_lang,
            target_lang=target_lang,
            timestamps=False # Toujours à False
        )
        
        text_result = output[0].text
        
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return (text_result,) # Retourne un tuple avec un seul élément


# --- Nodes ---

class CanaryModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "model_name": (get_canary_model_list(), ), } }
    RETURN_TYPES = ("CANARY_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Canary-ComfyUI"
    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("canary", model_name)
        if not model_path: raise FileNotFoundError(f"Model {model_name} not found.")
        if model_path in CACHED_MODELS:
            print(f"Canary-ComfyUI: Loading cached model '{model_name}'")
            return (CACHED_MODELS[model_path],)
        
        print(f"Canary-ComfyUI: Loading model '{model_name}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = ASRModel.restore_from(restore_path=model_path, map_location=device)
            model.to(device); model.eval()
            CACHED_MODELS[model_path] = model
            print(f"Canary-ComfyUI: Model '{model_name}' loaded successfully on {device}.")
            return (model,)
        except Exception as e:
            print(f"Canary-ComfyUI: An error occurred while loading the model: {e}")
            raise e

class CanaryASRNode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio": ("AUDIO",),
                "language": (ASR_LANGS, {"default": "en"}),
            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "Canary-ComfyUI"
    def process(self, canary_model, audio, language):
        return process_audio_and_run_model(canary_model, audio, language, language)

class CanaryTranslateToENNode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio": ("AUDIO",),
                "source_lang": (TRANSLATION_LANGS, {"default": "fr"}),
            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "Canary-ComfyUI"
    def process(self, canary_model, audio, source_lang):
        return process_audio_and_run_model(canary_model, audio, source_lang, 'en')
        
class CanaryTranslateFromENNode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio": ("AUDIO",),
                "target_lang": (TRANSLATION_LANGS, {"default": "fr"}),
            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "Canary-ComfyUI"
    def process(self, canary_model, audio, target_lang):
        return process_audio_and_run_model(canary_model, audio, 'en', target_lang)

# --- Mappings pour ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "CanaryModelLoader": CanaryModelLoader,
    "CanaryASRNode": CanaryASRNode,
    "CanaryTranslateToENNode": CanaryTranslateToENNode,
    "CanaryTranslateFromENNode": CanaryTranslateFromENNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CanaryModelLoader": "Load Canary Model",
    "CanaryASRNode": "Canary Transcription (ASR)",
    "CanaryTranslateToENNode": "Canary Translate to English (AST)",
    "CanaryTranslateFromENNode": "Canary Translate from English (AST)",
}