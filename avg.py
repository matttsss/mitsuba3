import numpy as np

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from helpers import render

res = (512, 1024)


if __name__ == "__main__":
    ## Plugin
    avg_sunsky = mi.load_dict({
        'type': 'avg_sunsky',
        'time_resolution': 500,
        'window_start_time': 6,
        'window_end_time': 18
    })

    # params = mi.traverse(avg_sunsky)
    # params['sun_scale'] = 0
    # params.update()

    image = render(avg_sunsky, res)
    mi.util.write_bitmap("results/test.exr", image)
