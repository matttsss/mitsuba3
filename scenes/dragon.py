from __future__ import annotations

import drjit as dr
import mitsuba as mi

def load_scene(render_size: int = 1024) -> tuple[mi.Scene, mi.SceneParameters]:
    T = mi.ScalarTransform4f

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'prb',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov': 20,
            'to_world': T().look_at(
                origin=[-40, 5, 50],
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
            'direction': mi.ScalarVector3f(0.1886, -0.6923, -0.6965),
            'irradiance': 10.0
        },
        # -------------------- Shapes --------------------
        'dragon': {
            'type': 'obj',
            'filename': 'resources/dragon/dragon.obj',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': 'resources/dragon/white.jpg'
                }
            }
        },
        'base': {
            'type': 'obj',
            'filename': 'resources/dragon/base.obj',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': 'resources/dragon/white.jpg'
                }
            }
        },
        'bigstone': {
            'type': 'obj',
            'filename': 'resources/dragon/bigstone.obj',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': 'resources/dragon/white.jpg'
                }
            }
        },
        'smallstone': {
            'type': 'obj',
            'filename': 'resources/dragon/smallstone.obj',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': 'resources/dragon/white.jpg'
                }
            }
        },
        'sword': {
            'type': 'obj',
            'filename': 'resources/dragon/sword.obj',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': 'resources/dragon/white.jpg'
                }
            }
        }
    }

    scene = mi.load_dict(scene_dict, optimize=False)
    scene_params = mi.traverse(scene)

    for key in scene_params.keys():
        if ".bsdf.reflectance.data" in key:
            dr.enable_grad(scene_params[key])

    scene_params.update()
    return scene, scene_params