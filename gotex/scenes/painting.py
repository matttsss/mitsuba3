from __future__ import annotations

import mitsuba as mi
from ..utils import resolve_texture_filename

def load_scene(texture_dir: str | None = None) -> dict:
    T = mi.ScalarTransform4f

    return {
        'type': 'scene',
        'integrator': {
            'type': 'direct',
            'bsdf_samples': 0,
            'emitter_samples': 1
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov': 35,
            'to_world': T().look_at(
                origin=[0, 0, 3],
                target=[0, 0, 0],
                up=[0, 0.000001, 1]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 128
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
        # 'background': {
        #     'type': 'constant',
        #     'radiance': 0.1
        # },

        'light': {
            'type': 'directional',
            'direction': mi.ScalarVector3f(0, 0, -1),
            'irradiance': 1.0
        },

        # -------------------- Shapes --------------------
        'background': {
            'type': 'rectangle',
            #'to_world': mi.ScalarAffineTransform4f.rotate([1, 0, 0], 90).scale([10, 1, 10]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'raw': True,
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'background',
                        'resources/noise_tex.exr'
                    ),
                    'format': 'variant'
                }
            }
        },
    }