from __future__ import annotations

import mitsuba as mi
from ..utils import resolve_texture_filename

def load_scene(texture_dir: str | None = None) -> dict:
    return {
        'type': 'scene',
        'integrator': {
            'type': 'prb',
            'max_depth': 8,
            'hide_emitters': True
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
        # -------------------- Light --------------------
        'light': {
            'type': 'directional',
            'direction': mi.ScalarVector3f(-0.6965, -0.6923, 0.1886),
            'irradiance': 10.0
        },
        # 'background': {
        #     'type': 'envmap',
        #     'filename': 'resources/dragon/envmap.exr',
        # },
        'background': {
            'type': 'constant',
            'radiance': 0.1
        },
        # -------------------- Shapes --------------------
        'dragon': {
            'type': 'ply',
            'filename': 'resources/dragon/dragon.ply',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'dragon',
                        'resources/dragon/large_blue_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        },
        'base': {
            'type': 'ply',
            'filename': 'resources/dragon/base.ply',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'base',
                        'resources/dragon/grey_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        },
        'bigstone': {
            'type': 'ply',
            'filename': 'resources/dragon/bigstone.ply',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'bigstone',
                        'resources/dragon/grey_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        },
        'smallstone': {
            'type': 'ply',
            'filename': 'resources/dragon/smallstone.ply',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'smallstone',
                        'resources/dragon/grey_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        },
        'sword': {
            'type': 'ply',
            'filename': 'resources/dragon/sword.ply',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'sword',
                        'resources/dragon/grey_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        }
    }