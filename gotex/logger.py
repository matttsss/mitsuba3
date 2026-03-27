from __future__ import annotations
import os

import torch
import mitsuba as mi

__out_dir__: str | None = None

def set_out_dir(out_dir: str):
    global __out_dir__
    __out_dir__ = out_dir

def save_config(config: dict, name: str = "config.yaml"):
    assert __out_dir__ is not None, "Output directory is not set. Call set_out_dir(out_dir) before saving config."

    with open(os.path.join(__out_dir__, name), 'w') as f:
        import yaml
        yaml.dump(config, f)

def save_image(image: mi.TensorXf, name: str):
    assert __out_dir__ is not None, "Output directory is not set. Call set_out_dir(out_dir) before saving images."
    assert image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in [3, 4]), "Expected image to have shape (H, W, C) or (H, W)"

    mi.util.write_bitmap(os.path.join(__out_dir__, name), image)

def save_tensor(tensor: mi.TensorXf | torch.Tensor, name: str):
    assert __out_dir__ is not None, "Output directory is not set. Call set_out_dir(out_dir) before saving tensors."
    assert tensor.ndim == 3, "Expected tensor to have shape (H, W, C)"

    save_path = os.path.join(__out_dir__, name)
    if isinstance(tensor, torch.Tensor):
        torch.save(tensor.detach().cpu(), save_path)
    elif isinstance(tensor, mi.TensorXf):
        mi.tensor_io.write(save_path, tensor)
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")