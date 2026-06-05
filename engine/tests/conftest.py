"""
Pytest configuration shared by all engine tests.

Adds the `engine/` directory to sys.path so `from core...` and `from plugins...`
imports work when tests are run from any cwd (the project root or engine/).
"""

import os
import sys

# Make the engine/ folder importable as a package root
_ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)
