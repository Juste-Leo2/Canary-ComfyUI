# __init__.py

# Importer les mappings des différents fichiers
from .nodes import NODE_CLASS_MAPPINGS as loader_nodes, NODE_DISPLAY_NAME_MAPPINGS as loader_display_nodes
from .src.Canary_1b_v2 import NODE_CLASS_MAPPINGS as canary_v2_nodes, NODE_DISPLAY_NAME_MAPPINGS as canary_v2_display_nodes
from .src.Canary_1b_flash import NODE_CLASS_MAPPINGS as canary_1b_flash_nodes, NODE_DISPLAY_NAME_MAPPINGS as canary_1b_flash_display_nodes
from .src.Canary_180m_flash import NODE_CLASS_MAPPINGS as canary_180m_flash_nodes, NODE_DISPLAY_NAME_MAPPINGS as canary_180m_flash_display_nodes

# Créer les dictionnaires finaux
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Fusionner les mappings de tous les fichiers
NODE_CLASS_MAPPINGS.update(loader_nodes)
NODE_CLASS_MAPPINGS.update(canary_v2_nodes)
NODE_CLASS_MAPPINGS.update(canary_1b_flash_nodes)
NODE_CLASS_MAPPINGS.update(canary_180m_flash_nodes)

NODE_DISPLAY_NAME_MAPPINGS.update(loader_display_nodes)
NODE_DISPLAY_NAME_MAPPINGS.update(canary_v2_display_nodes)
NODE_DISPLAY_NAME_MAPPINGS.update(canary_1b_flash_display_nodes)
NODE_DISPLAY_NAME_MAPPINGS.update(canary_180m_flash_display_nodes)


# Exporter les mappings finaux pour ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("Canary-ComfyUI: Loaded custom nodes for Canary-1B-v2, Canary-1B-Flash, and Canary-180m-Flash.")