from __future__ import annotations

import mitsuba as mi

from ..utils import resolve_texture_filename

def load_scene(texture_dir: str | None = None) -> dict:
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
                'width' : 1024,
                'height': 1024,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },

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
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'rubber_feet',
                        'resources/coffee_maker/black_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        },
        'black_inserts': {
            'type': 'ply',
            'filename': 'resources/coffee_maker/BlackInserts.ply',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'black_inserts',
                        'resources/coffee_maker/black_tex.png'
                    ),
                    'format': 'variant'
                }
            }
        },
        'black_joint': {
            'type': 'ply',
            'filename': 'resources/coffee_maker/BlackJoint.ply',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'black_joint',
                        'resources/coffee_maker/black_tex.png'
                    ),
                    'format': 'variant'
                }
            }
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
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'bitmap',
                    'filename': resolve_texture_filename(
                        texture_dir,
                        'yellow_cassing',
                        'resources/coffee_maker/orange_tex.png'
                    ),
                    'format': 'variant'
                }

            }
        },
    }