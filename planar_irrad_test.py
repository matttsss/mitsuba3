import drjit as dr
import mitsuba as mi


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    T = mi.ScalarAffineTransform4f

    area_light = {
            'type': 'rectangle',
            'to_world': T.translate([0, 2, 1]).rotate([1, 0, 0], 90).scale([2, 2, 1]),
            'emitter': {
                'type': 'area',
                'radiance': 0.001
            }
    }

    const_light = {
        'type': 'constant',
        'radiance': 0.001
    }

    timed_sunsky = {
        'type': 'timed_sunsky'
    }

    classic_sunsky = {
        'type': 'sunsky'
    }

    avg_envmap = {
        'type': 'envmap',
        'filename': 'results/average_1_5_aperture.exr',
        'scale': 0.005
    }

    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'planar_irradiancemeter',
            'sampler': {
                'type': 'independent',
                'sample_count': 1024
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 128,
                'height': 128,
                'rfilter': {
                    'type': 'box',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'shutter_close': 1.,
            'to_world': T.scale([5, 5, 1])
        },
        # -------------------- Light --------------------
        'light': timed_sunsky,
        # -------------------- Shapes --------------------
        'cube' : {
            'type': 'cube',
            'to_world': T.translate([0, 0, 1.])
        }
    }

    scene = mi.load_dict(scene)

    image = mi.render(scene)
    mi.util.write_bitmap("results/test.exr", image)
