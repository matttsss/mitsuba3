import numpy as np

import drjit as dr
import mitsuba as mi

## To be run on branch `avg_sunsky_debug_comp`

mi.set_variant("cuda_rgb_double")


def load_directions():
    import struct
    with open("results/directions.bin", "rb") as file:
        width = struct.unpack("I", file.read(4))[0]
        height = struct.unpack("I", file.read(4))[0]
        nb_rays = width * height

        x = mi.Float(struct.unpack(nb_rays * "f", file.read(4 * nb_rays)))
        y = mi.Float(struct.unpack(nb_rays * "f", file.read(4 * nb_rays)))
        z = mi.Float(struct.unpack(nb_rays * "f", file.read(4 * nb_rays)))
        return (width, height), mi.Vector3f(x, y, z)

        

def generate_rays(render_res):
    phis, thetas = dr.meshgrid(
        dr.linspace(mi.Float, 0.0, dr.two_pi, render_res[0], endpoint=False),
        dr.linspace(mi.Float, 0.0, dr.pi, render_res[1], endpoint=False)
    )

    sp, cp = dr.sincos(phis)
    st, ct = dr.sincos(thetas)

    si = mi.SurfaceInteraction3f()
    si.wi = -mi.Vector3f(cp*st, sp*st, ct)

    return si


def to_file(path, flat_image, res):
    image = mi.TensorXf(dr.ravel(flat_image), (*res[::-1], 3))
    image = mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32)
    mi.util.write_bitmap(path, image)


if __name__ == "__main__":
    hour = 12.3

    render_res = (2*255, 255)
    half_image_res = (render_res[0], (render_res[1]//2 - 1))

    point_average = mi.load_dict({
        "type": "avg_sunsky",
        "time_resolution": 1,
        "bitmap_height": 255,
        "end_year": 2025,
        "end_day": 2,
        "window_start_time": hour,
        "window_end_time": hour,
        "sun_scale": 0
    })

    res, rays = load_directions()
    dr.print(rays)

    sky = mi.load_dict({
        "type": "sunsky",
        "year": 2025,
        "month": 1,
        "day": 1,
        "hour": hour,
        "minute": 0,
        "second": 0,
        "sun_scale": 0
    })

    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = -rays

    point_average_res = point_average.eval(si)
    sky_res = sky.eval(si)

    to_file("results/sky.exr", sky_res, res)
