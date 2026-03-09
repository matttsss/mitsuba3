from __future__ import annotations

import numpy as np
import mitsuba as mi


def list_to_mat4f(lst: list[float]) -> mi.ScalarTransform4f:
    assert len(lst) == 16, "List must have 16 elements to convert to a 4x4 matrix"

    mat = np.array(lst, dtype=np.float32).reshape((4, 4))
    return mi.ScalarTransform4f(mat)
    

def get_light(to_world: mi.ScalarTransform4f) -> dict:
    return {
        'type': 'rectangle',
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.0, 0.0, 0.0]
            }
        },
        'emitter': {
            'type': 'area',
            'radiance': 2.0
        },
        'to_world': to_world
    }

def load_scene(render_size: int = 1024) -> tuple[mi.Scene, mi.SceneParameters, dict, dict]:
    T = mi.ScalarTransform4f

    glass_bsdf = {
        'type': 'dielectric',
        'int_ior': 1.5,
        'ext_ior': 1.0,
    }

    metal_bsdf = {
        'type': 'twosided',
        'bsdf': {
            'type': 'conductor',
            'material': 'none'
        }
    }

    floor_bsdf = {
        'type': 'twosided',
        'bsdf': {
            'type': 'plastic',
            'int_ior': 1.5,
            'ext_ior': 1.0,
            'diffuse_reflectance': {
                'type': 'rgb',
                'value': [0.578596, 0.578596, 0.578596]
            }
        }
    }

    diffuse_black_bsdf = {
        'type': 'diffuse',
        'reflectance': {
            'type': 'rgb',
            'value': [0.00631, 0.00631, 0.00631]
        }
    }

    if False:
        plastic_orange_bsdf = {
            'type': 'twosided',
            'bsdf': {
                'type': 'plastic',
                'int_ior': 1.5,
                'ext_ior': 1.0,
                'nonlinear': True,
                'diffuse_reflectance': {
                    'type': 'rgb',
                    'value': [1, 0.378676, 0.0134734]
                }
            }
        }

        plastic_black_bsdf = {
            'type': 'twosided',
            'bsdf': {
                'type': 'roughplastic',
                'int_ior': 1.5,
                'ext_ior': 1.0,
                'nonlinear': True,
                'alpha': 0.1,
                'distribution': 'ggx',
                'diffuse_reflectance': {
                    'type': 'rgb',
                    'value': [0.00631, 0.00631, 0.00631]
                }
            }
        }

    else:
        plastic_orange_bsdf = {
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'filename': 'resources/coffee_maker/orange_tex.png',
                'format': 'variant'
            }
        }

        plastic_black_bsdf = {
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'filename': 'resources/coffee_maker/black_tex.png',
                'format': 'variant'
            }
        }

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
            'fov': 35,
            'to_world': T().look_at(
                origin=[50, 5, -40],
                target=[0, 0.1, 0],
                up=[0, 1, 0]
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
        # 'light1': get_light(list_to_mat4f([
        #     0.0813859, -3.42425e-009, 0.0783378, -0.277923, 
        #     1.31768e-008, -0.224422, -1.25677e-008, 0.225271, 
        #     0.174532, 1.85401e-008, -0.0365296, 0.163724, 
        #     0, 0, 0, 1])),

        # 'light2': get_light(list_to_mat4f([
        #     -9.80906e-009, 0.00227401, -0.0820026, 0.322947, 
        #     0.224423, 1.56693e-016, -3.58473e-009, 0.25176, 
        #     -1.22092e-010, -0.182697, -0.00102068, 0.046278, 
        #     0, 0, 0, 1])),

        # 'light3': get_light(list_to_mat4f([
        #     -0.230128, -6.99084e-016, 1.59932e-008, 0, 
        #     -3.47484e-008, 4.62982e-009, -0.105918, 0.50385, 
        #     0, -0.230128, -1.00592e-008, 0.0372435, 
        #     0, 0, 0, 1])),

        'background': {
            'type': 'constant',
            'radiance': 0.1
        },

        'light': {
            'type': 'directional',
            'direction': mi.ScalarVector3f(-0.6965, -0.6923, 0.1886),
            'irradiance': 2.0
        },

        # -------------------- Shapes --------------------
        'background': {
            'type': 'rectangle',
            'to_world': mi.ScalarAffineTransform4f.rotate([1, 0, 0], 90).translate([0, -0.5, 0]).scale([10, 1, 10]),
            'bsdf': floor_bsdf
        },
        'rubber_feet': {
            'type': 'ply',
            'filename': 'resources/coffee_maker/RubberFeet.ply',
            'bsdf': diffuse_black_bsdf
        },
        'black_inserts': {
            'type': 'ply',
            'filename': 'resources/coffee_maker/BlackInserts.ply',
            'bsdf': diffuse_black_bsdf
        },
        'black_joint': {
            'type': 'ply',
            'filename': 'resources/coffee_maker/BlackJoint.ply',
            'bsdf': plastic_black_bsdf
        },
        'glass_bowl': {
            'type': 'ply',
            'filename': 'resources/coffee_maker/GlassBowl.ply',
            'bsdf': glass_bsdf
        },
        'metal_support': {
            'type': 'ply',
            'filename': 'resources/coffee_maker/MetalSupport.ply',
            'bsdf': metal_bsdf
        },
        'metal_pipes': {
            'type': 'ply',
            'filename': 'resources/coffee_maker/MetalPipes.ply',
            'bsdf': metal_bsdf
        },
        'yellow_cassing': {
            'type': 'ply',
            'filename': 'resources/coffee_maker/YellowCassing.ply',
            'bsdf': plastic_orange_bsdf
        },
    }

    scene = mi.load_dict(scene_dict, optimize=False)
    scene_params = mi.traverse(scene)

    sd_config = dict(
        prompt="A photo of a coffee maker, with a yellow casing, metal pipes and support, a glass bowl",
        negative_prompt="",
        cn_cond_scale=0.6,
        render_size=render_size,
        guidance_scale=50.0,
        min_time=0.02,
        max_time=0.98
    )

    scene_metadata = {
        'scene_name': 'coffee_maker',
        'is_2d': False,
        'target': mi.ScalarVector3f(0, 0.2, 0),
        'radius': 1
    }

    return scene, scene_params, scene_metadata, sd_config