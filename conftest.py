"""Asegura que `import src...` funcione al correr pytest desde la raíz del proyecto."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
