import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

tpe = "tgmm"
spp = 4096
scene: mi.Scene = mi.load_dict({
    "type": "scene",
    "integrator": {
        "type": "direct",
        "bsdf_samples": 1,
        "emitter_samples": 1
    },
    "env": {
        "type": "timed_sunsky",
        "sun_scale": 1
    },
    "plane": {
        'type': 'sphere',
        'bsdf': {
            'type': 'diffuse',
        }
    },
    "sensor": {
        "type": "perspective",
        "shutter_close": 1,
        "to_world": mi.ScalarAffineTransform4f.look_at(
            mi.ScalarPoint3f(2, 5, 0.3), 
            mi.ScalarPoint3f(0), 
            mi.ScalarPoint3f(0, 0, 1)
        ),
        "sampler": {
            "type": "independent",
            "sample_count": spp
        },
        "film": {
            "type": "hdrfilm",
            "width": 512,
            "height": 512,
            "filter": {
                "type": "box"
            }
        }
    }
})

for i in range(0):
    image = mi.render(scene)
    dr.eval(image)

image = mi.render(scene)
mi.util.write_bitmap(f"sampling_tests/out/{tpe}_out_{spp}.exr", image)