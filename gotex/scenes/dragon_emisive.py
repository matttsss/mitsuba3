from __future__ import annotations

import mitsuba as mi
from ..utils import resolve_texture_filename

def load_scene(texture_dir: str | None = None) -> dict:
    return {
        'type': 'scene',
        'integrator': {
            'type': 'direct',
            'bsdf_samples': 0
        },
        # -------------------- Sensor --------------------
        'sensor': {

            'type': 'perspective',
            'fov': 30,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[50, 5, -40],
                target=[0, 10, 0],
                up=[0, 1, 0]
            ),

            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 1024,
                'height': 1024,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # -------------------- Shapes --------------------
        'dragon': {
            'type': 'ply',
            'filename': 'resources/dragon/dragon.ply',
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'dragon_emissive',
                        'resources/dragon/blue_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        },
        'base': {
            'type': 'ply',
            'filename': 'resources/dragon/base.ply',
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'dragon_emissive',
                        'resources/dragon/grey_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        },
        'bigstone': {
            'type': 'ply',
            'filename': 'resources/dragon/bigstone.ply',
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'dragon_emissive',
                        'resources/dragon/grey_tex.png'
                    ),
                    'format': 'variant'
                }
            }

        },
        'smallstone': {
            'type': 'ply',
            'filename': 'resources/dragon/smallstone.ply',
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'dragon_emissive',
                        'resources/dragon/grey_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        },
        'sword': {
            'type': 'ply',
            'filename': 'resources/dragon/sword.ply',
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'dragon_emissive',
                        'resources/dragon/grey_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        }
    }