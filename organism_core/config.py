"""Workspace-local paths; no dependency on a maintainer's computer."""
import os
from pathlib import Path
SOURCE_ROOT=Path(__file__).resolve().parents[1]
HOME=Path(os.environ.get('FLY_EFFECT_HOME',Path.cwd())).expanduser().resolve()
DATA=Path(os.environ.get('FLY_EFFECT_DATA_DIR',HOME/'data')).expanduser().resolve()
GRAPH=DATA/'graph'

os.environ.setdefault('FLYGYM_ASSET_CACHE_DIR',str(HOME/'cache/flygym-assets'))
