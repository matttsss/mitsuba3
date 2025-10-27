import drjit as dr
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")


def render(complex_sun, sun_dir):
    scene = {
        "type": "scene",
        "integrator": {
            'type': 'direct'
        },
        "sensor": {
            "type": "perspective",
            "fov": 0.8,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0],
                target=sun_dir,
                up=[0, 0, 1]
            ),
            "film": {
                "type": "hdrfilm",
                "width": 512,
                "height": 512,
                "rfilter": {"type": "gaussian"}
            }
        },
        "emitter": {
            "type": "sunsky",
            "sun_direction": sun_dir,
            "complex_sun": complex_sun
        }

    }

    scene = mi.load_dict(scene)

    image = mi.render(scene, spp=256)
    mi.util.write_bitmap(f"results/render_sun_{'complex' if complex_sun else 'simple'}.exr", image)

sun_eta = dr.deg2rad(50)
sun_phi = 0.5 * dr.pi

sp, cp = dr.sincos(sun_phi)
st, ct = dr.sincos(0.5 * dr.pi - sun_eta)

sun_dir = mi.ScalarVector3f(sp * st, cp * st, ct)

render(True, sun_dir)
render(False, sun_dir)
