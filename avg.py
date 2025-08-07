import numpy as np

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from helpers import render, generate_average

res = (512, 1024)


if __name__ == "__main__":


    if True:
    
        timed_sunsky = mi.load_dict({
            'type': 'timed_sunsky',
            'window_start_time': 6,
            'window_end_time': 18,
            'sun_aperture': 1.5,
        })

        image = generate_average(timed_sunsky, res, 400)
        image = mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32)
        mi.util.write_bitmap("results/test.exr", image)

    else:

        ## Plugin
        avg_sunsky = mi.load_dict({
            'type': 'avg_sunsky',
            'time_samples_per_day': 400,
            'window_start_time': 6,
            'window_end_time': 18,
            'sun_aperture': 1.5,
        })
        image = render(avg_sunsky, res)
        image = mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32)
        mi.util.write_bitmap("results/test.exr", image)