from __future__ import annotations

import mitsuba as mi

def load_scene(render_size: int = 1024) -> tuple[mi.Scene, mi.SceneParameters, dict, dict]:
    T = mi.ScalarTransform4f

    scene_dict = {
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
            'to_world': T().look_at(
                origin=[50, 5, -40],
                target=[0, 10, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 32
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
                    'filename': 'resources/dragon/blue_tex.png',
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
                    'filename': 'resources/dragon/grey_tex.png',
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
                    'filename': 'resources/dragon/grey_tex.png',
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
                    'filename': 'resources/dragon/grey_tex.png',
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
                    'filename': 'resources/dragon/grey_tex.png',
                    'format': 'variant'
                }
            }
        }
    }

    sd_config = dict(
        prompt="A photo of a dragon with blue scale on a rocky piedestal, under a directional light",
        negative_prompt="unrealistic, blurry, low quality, oversaturation.",
        cn_cond_scale=0.6,
        render_size=render_size,
        guidance_scale=50.0,
        min_time=0.02,
        max_time=0.98
    )

    scene_metadata = {
        'scene_name': 'dragon',
        'is_2d': False,
        'target': mi.ScalarVector3f(0, 7, 0),
        'radius': 60,
    }

    scene = mi.load_dict(scene_dict, optimize=False)
    scene_params = mi.traverse(scene)
    return scene, scene_params, scene_metadata, sd_config