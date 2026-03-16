from __future__ import annotations

import mitsuba as mi
from utils import resolve_texture_filename

def load_scene(
    render_size: int = 1024,
    nb_sensors: int = 6,
    texture_dir: str | None = None
) -> dict:

    return {
        'camera_config': {
            'is_2d': False,
            'target': mi.ScalarPoint3f(0, 7, 0),
            'radius_min': 55,
            'radius_max': 65,
            'elevation_min': -10,
            'elevation_max': 80
        },

        'scene_name': 'dragon',
        'scene': {
            'type': 'scene',
            'integrator': {
                'type': 'prb',
                'max_depth': 8,
                'hide_emitters': True
            },
            # -------------------- Sensor --------------------
            'sensor': {
                'type': 'batch',

                **{
                    f'sensor_{i}': {
                        'type': 'perspective',
                        'fov': 30,
                        'to_world': mi.ScalarTransform4f()
                    } for i in range(nb_sensors)
                },

                'sampler': {
                    'type': 'independent',
                    'sample_count': 16
                },
                'film': {
                    'type': 'hdrfilm',
                    'width' : nb_sensors * render_size,
                    'height': render_size,
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
                            'resources/dragon/blue_tex.png'
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
        },
        'sd_config': {
            'prompt': "A photo of a blue dragon on a sandypiedestal, under a directional light",
            'negative_prompt': "unrealistic, blurry, low quality, oversaturation.",
            'cn_cond_scale': 0.6,
            'render_size': render_size,
            'guidance_scale': 50.0,
            'min_time': 0.02,
            'max_time': 0.98
        }
    }