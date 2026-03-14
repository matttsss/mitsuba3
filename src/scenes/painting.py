from __future__ import annotations

import mitsuba as mi

def load_scene(render_size: int = 1024) -> tuple[mi.Scene, mi.SceneParameters, dict, dict]:
    T = mi.ScalarTransform4f

    scene_dict = {
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
                    'filename': 'resources/noise_tex.exr',
                    'format': 'variant'
                }
            }
        },
    }
    
    scene_config = {
        'scene_name': 'painting',
        'scene': scene_dict,
        'camera_config': {
            'is_2d': True,
        },
        'sd_config': {
            'prompt': "A DSLR image of a hamburger",
            'negative_prompt': "",
            'cn_cond_scale': 0.0,
            'render_size': render_size,
            'guidance_scale': 50.0,
            'min_time': 0.02,
            'max_time': 0.98
        },
    }

    return scene_config