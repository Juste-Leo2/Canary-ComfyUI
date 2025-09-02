# src/Canary_180m_flash.py

import os
import torch
import tempfile
import numpy as np
import torchaudio
from scipy.io.wavfile import write as write_wav

# --- Constantes Spécifiques au Modèle Canary-180m-Flash ---
ASR_LANGS_FLASH = sorted(["en", "de", "fr", "es"])
TRANSLATION_LANGS_FLASH = sorted([lang for lang in ASR_LANGS_FLASH if lang != 'en'])


# --- Fonction de Traitement Spécifique ---

def process_audio_with_canary_180m_flash(model, audio, source_lang, target_lang):
    """
    Fonction helper qui prépare l'audio et exécute le modèle Canary-180m-Flash.
    L'option PnC est retirée car elle posait problème avec ce modèle.
    """
    # Configuration spécifique, appliquée une seule fois
    if not hasattr(model, '_180m_flash_configured'):
        print("Canary-ComfyUI: First time setup for Canary-180m-Flash model, setting beam_size=1.")
        decode_cfg = model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decode_cfg)
        model._180m_flash_configured = True

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
        
        print(f"Canary-ComfyUI: Running {task_name} for Canary-180m-Flash (source: {source_lang}, target: {target_lang}) on device: {device}")
        
        output = model.transcribe(
            [audio_path],
            source_lang=source_lang,
            target_lang=target_lang,
            pnc='no',
            timestamps='no'
        )
        
        text_result = output[0].text
        
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return (text_result,)


# --- Nœuds Spécifiques au Modèle Canary-180m-Flash ---

class Canary180mFlashASRNode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio": ("AUDIO",),
                "language": (ASR_LANGS_FLASH, {"default": "en"}),
                # L'option "pnc" a été retirée de l'interface
            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "Canary-ComfyUI/Canary-180m-Flash"
    def process(self, canary_model, audio, language):
        return process_audio_with_canary_180m_flash(canary_model, audio, language, language)

class Canary180mFlashTranslateToENNode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio": ("AUDIO",),
                "source_lang": (TRANSLATION_LANGS_FLASH, {"default": "fr"}),
            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "Canary-ComfyUI/Canary-180m-Flash"
    def process(self, canary_model, audio, source_lang):
        return process_audio_with_canary_180m_flash(canary_model, audio, source_lang, 'en')
        
class Canary180mFlashTranslateFromENNode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "canary_model": ("CANARY_MODEL",),
                "audio": ("AUDIO",),
                "target_lang": (TRANSLATION_LANGS_FLASH, {"default": "fr"}),
            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "Canary-ComfyUI/Canary-180m-Flash"
    def process(self, canary_model, audio, target_lang):
        return process_audio_with_canary_180m_flash(canary_model, audio, 'en', target_lang)

# --- Mappings pour les nœuds de CE fichier ---
NODE_CLASS_MAPPINGS = {
    "Canary180mFlashASRNode": Canary180mFlashASRNode,
    "Canary180mFlashTranslateToENNode": Canary180mFlashTranslateToENNode,
    "Canary180mFlashTranslateFromENNode": Canary180mFlashTranslateFromENNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Canary180mFlashASRNode": "Canary-180m-Flash Transcription (ASR)",
    "Canary180mFlashTranslateToENNode": "Canary-180m-Flash Translate to EN (AST)",
    "Canary180mFlashTranslateFromENNode": "Canary-180m-Flash Translate from EN (AST)",
}