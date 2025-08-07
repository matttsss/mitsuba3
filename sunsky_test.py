import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

from helpers import render

sunsky = mi.load_dict({
    'type': 'sunsky'
})

image = render(sunsky, (512, 1024))
mi.util.write_bitmap("results/sunsky_test.exr", mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32))