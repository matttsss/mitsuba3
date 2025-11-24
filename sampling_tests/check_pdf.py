import numpy as np
import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')


resolution = (256, 512)

def test_and_save(plugin, si, ds, filename):
    radiance = plugin.eval(si)
    luminance = mi.luminance(radiance)

    pdf = plugin.pdf_direction(mi.Interaction3f(), ds)

    luminance = mi.TensorXf(luminance, shape=resolution)
    pdf = mi.TensorXf(pdf, shape=resolution)

    mi.util.write_bitmap(f'sampling_tests/out/{filename}_lum.exr', luminance)
    mi.util.write_bitmap(f'sampling_tests/out/{filename}_pdf.exr', pdf)


plugin = mi.load_dict({
    'type': 'timed_sunsky',
    'end_year': 2025,
    'sun_scale': 0
})

theta, phi = dr.meshgrid(
    dr.linspace(mi.Float, 0, dr.pi / 2, resolution[0]),
    dr.linspace(mi.Float, 0, 2 * dr.pi, resolution[1]),
    indexing='ij'
)

view_dirs = mi.Vector3f(
    dr.sin(theta) * dr.cos(phi),
    dr.sin(theta) * dr.sin(phi),
    dr.cos(theta)
)

si = dr.zeros(mi.SurfaceInteraction3f)
si.wi = -view_dirs

ds = dr.zeros(mi.DirectionSample3f)
ds.d = -si.wi

dr.eval(si, ds)

for t in np.linspace(0, 1, 10):
    si.time = mi.Float(t)
    ds.time = mi.Float(t)

    test_and_save(plugin, si, ds, f"check_timed_sunsky_{t:.2f}")


plugin = mi.load_dict({
    'type': 'sunsky',
    'sun_scale': 0,
    'turbidity': 5.0,
    'sun_direction': dr.normalize(mi.ScalarVector3f(0.25, 0.3, 0.6))
})

test_and_save(plugin, si, ds, "check_sunsky")
