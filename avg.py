import numpy as np

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from helpers import generate_average

res = (256, 512)


if __name__ == "__main__":

    avg_sunsky = mi.load_dict({
        'type': 'timed_sunsky',
        'window_start_time': 6,
        'window_end_time': 18,
        #'sun_aperture': 1.5
    })

    dr.print(avg_sunsky)

    image = generate_average(avg_sunsky, res, 500)
    mi.util.write_bitmap("results/timed_avg_dump.exr", image)
