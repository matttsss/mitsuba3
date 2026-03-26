from __future__ import annotations

import mitsuba as mi
from ..utils import resolve_texture_filename

def load_scene(render_size: int = 1024, texture_dir: str | None = None) -> dict:
    return {
        'camera_config': {
            'is_2d': False,
            'target': mi.ScalarPoint3f(0, 7, 0),
            'radius_min': 55,
            'radius_max': 65,
            'elevation_min': -10,
            'elevation_max': 60,

            'camera_perturb': 0.05,
            'center_perturb': 0.1,
            'up_perturb': 0.02
        },

        'scene_name': 'dragon',
        'scene': {
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
                    'width' : render_size,
                    'height': render_size,
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
        },
        'sd_config': {
            'prompt': "A photo of a realistic blue dragon on a plain white piedestal, wings deployed above his head",
            'negative_prompt': "unrealistic, blurry, low quality, oversaturation.",
            'cn_cond_scale': 0.0,
            'render_size': render_size,
            'guidance_scale': 100.0,
            'min_time': 0.3,
            'max_time': 0.98
        }
    }