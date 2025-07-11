import numpy as np

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

N = 200
res = (512, 1024)

## Rays
phis, thetas = dr.meshgrid(
    dr.linspace(mi.Float, 0.0, dr.two_pi, res[1]),
    dr.linspace(mi.Float, dr.pi, 0.0, res[0]))
sp, cp = dr.sincos(phis)
st, ct = dr.sincos(thetas)

si = dr.zeros(mi.SurfaceInteraction3f)
si.wi = mi.Vector3f(cp*st, sp*st, ct)

## Plugin
sunsky = mi.load_dict({
    'type': 'avg_sunsky'
})


# @dr.syntax
# def gen_avg(plugin, dir, N):
#     start_time, end_time = 9, 16
#     dt = (end_time - start_time) / N
#     sunsky_params = mi.traverse(plugin)
# 
#     result = dr.zeros(mi.Color3f, dr.prod(res))
# 
#     i = mi.Int(0)
#     while mi.Bool(i < N):
# 
#         sunsky_params["hour"] = start_time + dt * i
#         sunsky_params.update()
# 
#         result += sunsky.eval(dir)
#         i += 1
# 
#     result /= N
# 
#     return result

def gen_avg(plugin, dir, N):
    start_time, end_time = 9, 16
    dt = (end_time - start_time) / N
    plugin_params = mi.traverse(plugin)

    result = dr.zeros(mi.Color3f, dr.prod(res))

    i = 0
    while i < N:
        plugin_params["hour"] = start_time + i*dt
        plugin_params.update()

        result += plugin.eval(dir)
        i += 1

    result /= N

    return result

avg_ray = gen_avg(sunsky, si, N)
image = mi.TensorXf(dr.ravel(sunsky.eval(si)), (*res, 3))
mi.util.write_bitmap("results/test.exr", image)
