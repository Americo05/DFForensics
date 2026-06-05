"""
Canonical plugin name constants.

Why this exists
---------------
SCENE_PLUGIN_WEIGHTS (in scene_classifier.py) and the cloud-mode filter
(in plugin_manager.py) reference plugins by their `plugin_name` string.
If a plugin renames itself, those tables silently stop matching and the
plugin is excluded from the pipeline with no error — scores just drop
its contribution to zero.

Forcing both the plugins AND the routing tables to import the same
constant from here means renaming a plugin requires editing this file,
which then either breaks imports loudly or stays consistent across
both sides.

When adding a new plugin
------------------------
1. Add its canonical name constant here.
2. Use it as the return value of the plugin's `plugin_name` property.
3. Reference it in SCENE_PLUGIN_WEIGHTS if you want it routed.
"""

from typing import Final


class PluginNames:
    """Canonical, frozen names for every detector plugin."""

    VIT_DETECTOR: Final[str]      = "Deepfake-Specific ViT Detector"
    DCT_FREQUENCY: Final[str]     = "DCT Frequency Analyzer"
    EDGE_BLENDING: Final[str]     = "Edge Blending Boundary Detector"
    SIGHTENGINE_CLOUD: Final[str] = "Sightengine Cloud Detector"
    PRNU_NOISE: Final[str]        = "PRNU Noise Residue Detector"
    MESONET: Final[str]           = "MesoNet (Afchar et al., WIFS 2018)"
