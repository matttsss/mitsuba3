from __future__ import annotations

import mitsuba as mi

def load_scene(render_size: int = 1024) -> tuple[mi.Scene, mi.SceneParameters]:
    T = mi.ScalarTransform4f

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'prb',
            'max_depth': 8,
            #'hide_emitters': True
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
                'sample_count': 64
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
        'background': {
            'type': 'envmap',
            'filename': 'resources/dragon/envmap.exr',
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

    scene = mi.load_dict(scene_dict, optimize=False)
    scene_params = mi.traverse(scene)
    return scene, scene_params