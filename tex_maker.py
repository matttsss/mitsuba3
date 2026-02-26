import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

avg_color = mi.ScalarVector3f(0.2, 0.2, 0.2)

rng = dr.rng(seed=0)
texture = rng.normal(mi.Float, shape=dr.prod((1024, 1024, 3)))

idx = dr.arange(mi.UInt32, 1024 * 1024) * 3
dr.scatter(texture, 0.005 * dr.gather(mi.Float, texture, idx + 0) + avg_color[0], idx + 0)
dr.scatter(texture, 0.005 * dr.gather(mi.Float, texture, idx + 1) + avg_color[1], idx + 1)
dr.scatter(texture, 0.005 * dr.gather(mi.Float, texture, idx + 2) + avg_color[2], idx + 2)

texture = dr.clip(texture, 0, 1)

texture = dr.reshape(mi.TensorXf, texture, (1024, 1024, 3))

mi.util.write_bitmap('resources/dragon/grey_tex.png', texture)
