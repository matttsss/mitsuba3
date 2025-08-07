import numpy as np
import drjit as dr
import mitsuba as mi

UINT_32_MAX = (1 << 32) - 1

def get_spherical_rays_scalar(image_res):
    phis, thetas = np.meshgrid(
        np.linspace(0.0, 2 * np.pi, image_res[1], endpoint=False),
        np.linspace(np.pi, 0.0, image_res[0], endpoint=False)
    )
    phis, thetas = np.ravel(phis), np.ravel(thetas)
    return [np.cos(phis) * np.sin(thetas), np.sin(phis) * np.sin(thetas), np.cos(thetas)]

def get_spherical_rays(image_res):
    phis, thetas = dr.meshgrid(
        dr.linspace(mi.Float, 0.0, dr.two_pi, image_res[1], endpoint=False),
        dr.linspace(mi.Float, dr.pi, 0.0, image_res[0], endpoint=False)
    )
    sp, cp = dr.sincos(phis)
    st, ct = dr.sincos(thetas)
    return mi.Vector3f(cp*st, sp*st, ct)

def render(plugin, resolution):
    si = dr.zeros(mi.SurfaceInteraction3f)

    if "scalar" in mi.variant():
        rays = get_spherical_rays_scalar(resolution)
        rays = np.transpose(rays)
        image = []
        for ray in rays:
            si.wi = mi.Vector3f(ray)
            image.append(plugin.eval(si))
        return mi.TensorXf(np.ravel(image), (*resolution, 3))
    
    else:
        si.wi = get_spherical_rays(resolution)
        colors = plugin.eval(si)
        return mi.TensorXf(dr.ravel(colors), (*resolution, 3))

def generate_average(plugin, render_res, day_resolution):
    nb_rays = dr.prod(render_res)
    nb_time_samples = day_resolution * 365 # TODO count for leap years
    pixel_idx = dr.arange(mi.UInt32, nb_rays)

    rays = get_spherical_rays(render_res)

    times = dr.linspace(mi.Float, 0.0, 1.0, nb_time_samples, endpoint=False)

    time_width = UINT_32_MAX // nb_rays
    time_width = dr.minimum(time_width, nb_time_samples)

    if time_width * nb_rays > UINT_32_MAX:
        time_width -= 1

    time_idx = dr.arange(mi.UInt32, time_width)

    pixel_idx_wav, time_idx = dr.meshgrid(pixel_idx, time_idx)

    result = dr.zeros(mi.Color3f, nb_rays)
    si = dr.zeros(mi.SurfaceInteraction3f, nb_rays * time_width)
    for frame_start in range(0, nb_time_samples, time_width):
        time_idx_wav = time_idx + frame_start
        active = time_idx_wav < nb_time_samples

        si.wi = dr.gather(mi.Vector3f, rays, pixel_idx_wav, active)
        si.time = dr.gather(mi.Float, times, time_idx_wav, active)

        color = plugin.eval(si, active) / nb_time_samples
        dr.scatter_add(result, color, pixel_idx_wav, active)


    return mi.TensorXf(dr.ravel(result), (*render_res, 3))