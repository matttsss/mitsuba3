import numpy as np

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from helpers import render

res = (255, 512)


if __name__ == "__main__":

    point_average = mi.load_dict({
        "type": "timed_sunsky",
        "albedo": mi.ScalarColor3f(0, 0, 1),
        "end_year": 2025,
        "end_day": 2,
        "window_start_time": 10,
        "window_end_time": 10,
        "sun_aperture": 10,
        "sun_scale": 1,
    })

    sky = mi.load_dict({
        "type": "sunsky",
        "albedo":  mi.ScalarColor3f(0, 0, 1),
        "year": 2025,
        "month": 1,
        "day": 1,
        "hour": 10,
        "minute": 0,
        "second": 0,
        "sun_aperture": 10,
        "sun_scale": 1,
    })

    
    point_average_image = render(point_average, res)
    sky_image = render(sky, res)

    mi.util.write_bitmap("results/point_average.exr", point_average_image)
    mi.util.write_bitmap("results/sky.exr", sky_image)
