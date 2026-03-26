import os, warnings, logging, torch
from pathlib import Path

import importlib
import mitsuba as mi

from diffusers.utils import logging as diffusers_logging
from huggingface_hub.utils import logging as hf_hub_logging
from transformers.utils import logging as transformers_logging

def silence_hf_warnings() -> None:
    # Keep runtime output readable by hiding known non-actionable third-party warnings.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    warnings.filterwarnings(
        "ignore",
        message=r"The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`.*",
        category=UserWarning,
    )
    transformers_logging.set_verbosity_error()
    diffusers_logging.set_verbosity_error()
    hf_hub_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def resolve_texture_filename(
    texture_dir: str | Path | None,
    object_name: str,
    default_filename: str
) -> str:
    if texture_dir is None:
        return default_filename
    
    texture_dir = Path(texture_dir) if texture_dir is not None else None

    for ext in ('.exr', '.png', '.jpg', '.jpeg', '.hdr', '.tif', '.tiff'):
        candidate = texture_dir / f'{object_name}{ext}'
        if candidate.exists():
            return str(candidate)

    return default_filename

def hdr_to_sdr(img, exposure=1.0):
    # 1. Apply exposure
    img = img * exposure
    
    # 2. Tone mapping (ACES)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    img = (img*(a*img+b)) / (img*(c*img+d)+e)
    
    # 3. Clamp
    img = torch.clamp(img, 0, 1)
    
    # 4. Gamma correction
    img = torch.where(
        img <= 0.0031308,
        12.92 * img,
        1.055 * torch.pow(img, 1/2.4) - 0.055
    )
    
    return img

def load_scene(path: str | None, texture_dir_override: str | None = None):
    """Load a custom scene with Stable Diffusion configuration"""
    
    # Dynamically import load_scene from the provided path
    if path is None:
        # Use default scene
        from .scenes.painting import load_scene
    else:
        scene_path = Path(path)
        if scene_path.suffix == '.py':
            # It's a file path, load as module
            spec = importlib.util.spec_from_file_location("custom_scene", scene_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            load_scene = module.load_scene
        else:
            # Assume it's a module path like "scenes.painting" or "scenes.dragon"
            path = "gotex.scenes." + path
            module = importlib.import_module(path)
            load_scene = module.load_scene
        
    scene_config = load_scene(texture_dir=texture_dir_override)
    scene = mi.load_dict(scene_config, optimize=False)

    return scene