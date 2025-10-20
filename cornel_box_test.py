import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

cb = mi.cornell_box()
cb.pop("ceiling")
cb.pop("light")
cb.pop("green-wall")
cb.pop("red-wall")
cb.pop("back")

cb["sunsky"] = {
    "type": "sunsky",
    "complex_sun": True,
    "to_world": mi.ScalarAffineTransform4f.rotate([1, 0, 0], -90).rotate([0, 0, 1], 50),
}

cb = mi.load_dict(cb)

render = mi.render(cb, spp=48)
mi.util.write_bitmap('results/cornell_box_test_old.exr', render)