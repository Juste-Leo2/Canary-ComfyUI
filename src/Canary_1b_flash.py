# src/Canary_1b_flash.py

import os
import torch
import tempfile
import numpy as np
import torchaudio
from scipy.io.wavfile import write as write_wav

# --- Constantes Spécifiques au Modèle Canary-1B-Flash ---
ASR_LANGS_FLASH = sorted(["en", "de", "fr", "es"])
TRANSLATION_LANGS_FLASH = sorted([lang for lang in ASR_LANGS_FLASH if lang != 'en'])


# --- Fonction de Traitement Spécifique ---

def process_audio_with_canary_flash(model, audio, source_lang, target_lang, pnc):
    """
    Fonction helper qui prépare l'audio et exécute le modèle Canary-1B-Flash.
    Retourne uniquement le texte résultant.
    """
    # Configuration spécifique au modèle flash, appliquée une seule fois
    if not hasattr(model, '_flash_configured'):
        print("Canary-ComfyUI: First time setup for Canary-Flash model, setting beam_size=1.")
        decode_cfg = model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decode_cfg)
        model._flash_configured = True

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
        print(f"Canary-ComfyUI: Running {task_name} for Canary-1B-Flash (source: {source_lang}, target: {target_lang}, PnC: {pnc}) on device: {device}")
        
        output = model.transcribe(
            [audio_path],
            source_lang=source_lang,
            target_lang=target_lang,
            pnc=pnc, # Nouvelle option pour la ponctuation
            timestamps='no' # Timestamps désactivés pour garder une sortie simple (STRING)
        )
        
        text_result = output[0].text
        
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return (text_result,)


# --- Nœuds Spécifiques au Modèle Canary-1B-Flash ---

class CanaryFlashASRNode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio": ("AUDIO",),
                "language": (ASR_LANGS_FLASH, {"default": "en"}),
                "pnc": (["yes", "no"], {"default": "yes"}),
            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "Canary-ComfyUI/Canary-1B-Flash"
    def process(self, canary_model, audio, language, pnc):
        return process_audio_with_canary_flash(canary_model, audio, language, language, pnc)

class CanaryFlashTranslateToENNode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio": ("AUDIO",),
                "source_lang": (TRANSLATION_LANGS_FLASH, {"default": "fr"}),
                "pnc": (["yes", "no"], {"default": "yes"}),
            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "Canary-ComfyUI/Canary-1B-Flash"
    def process(self, canary_model, audio, source_lang, pnc):
        return process_audio_with_canary_flash(canary_model, audio, source_lang, 'en', pnc)
        
class CanaryFlashTranslateFromENNode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio": ("AUDIO",),
                "target_lang": (TRANSLATION_LANGS_FLASH, {"default": "fr"}),
                "pnc": (["yes", "no"], {"default": "yes"}),
            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "Canary-ComfyUI/Canary-1B-Flash"
    def process(self, canary_model, audio, target_lang, pnc):
        return process_audio_with_canary_flash(canary_model, audio, 'en', target_lang, pnc)

# --- Mappings pour les nœuds de CE fichier ---
# NOTE: J'ai renommé en "NODE_CLASS_MASS_MAPPINGS" pour corriger une faute de frappe dans votre __init__.py
NODE_CLASS_MAPPINGS = {
    "CanaryFlashASRNode": CanaryFlashASRNode,
    "CanaryFlashTranslateToENNode": CanaryFlashTranslateToENNode,
    "CanaryFlashTranslateFromENNode": CanaryFlashTranslateFromENNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CanaryFlashASRNode": "Canary-1B-Flash Transcription (ASR)",
    "CanaryFlashTranslateToENNode": "Canary-1B-Flash Translate to EN (AST)",
    "CanaryFlashTranslateFromENNode": "Canary-1B-Flash Translate from EN (AST)",
}