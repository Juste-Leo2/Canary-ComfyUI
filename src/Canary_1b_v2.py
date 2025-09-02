# src/Canary_1b_v2.py

import os
import torch
import tempfile
import numpy as np
import torchaudio
from scipy.io.wavfile import write as write_wav

# --- Constantes Spécifiques au Modèle Canary-1B ---
# Ces langues sont supportées par le modèle Canary-1B
ASR_LANGS = sorted(["en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"])
TRANSLATION_LANGS = sorted([lang for lang in ASR_LANGS if lang != 'en'])


# --- Fonction de Traitement Spécifique ---

def process_audio_with_canary_1b(model, audio, source_lang, target_lang):
    """
    Fonction helper qui prépare l'audio et exécute le modèle Canary-1B.
    Retourne uniquement le texte résultant.
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
        print(f"Canary-ComfyUI: Running {task_name} for Canary-1B (source: {source_lang}, target: {target_lang}) on device: {device}")
        
        output = model.transcribe(
            [audio_path],
            source_lang=source_lang,
            target_lang=target_lang,
            timestamps=False # Toujours à False pour ce modèle
        )
        
        text_result = output[0].text
        
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return (text_result,)


# --- Nœuds Spécifiques au Modèle Canary-1B ---

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
    CATEGORY = "Canary-ComfyUI/Canary-1B"
    def process(self, canary_model, audio, language):
        return process_audio_with_canary_1b(canary_model, audio, language, language)

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
    CATEGORY = "Canary-ComfyUI/Canary-1B"
    def process(self, canary_model, audio, source_lang):
        return process_audio_with_canary_1b(canary_model, audio, source_lang, 'en')
        
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
    CATEGORY = "Canary-ComfyUI/Canary-1B"
    def process(self, canary_model, audio, target_lang):
        return process_audio_with_canary_1b(canary_model, audio, 'en', target_lang)

# --- Mappings pour les nœuds de CE fichier ---
NODE_CLASS_MAPPINGS = {
    "CanaryASRNode": CanaryASRNode,
    "CanaryTranslateToENNode": CanaryTranslateToENNode,
    "CanaryTranslateFromENNode": CanaryTranslateFromENNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CanaryASRNode": "Canary-1B Transcription (ASR)",
    "CanaryTranslateToENNode": "Canary-1B Translate to English (AST)",
    "CanaryTranslateFromENNode": "Canary-1B Translate from English (AST)",
}