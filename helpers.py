import numpy as np
import drjit as dr
import mitsuba as mi


def get_spherical_rays_scalar(image_res):
    phis, thetas = np.meshgrid(
        np.linspace(0.0, 2 * np.pi, image_res[1]),
        np.linspace(np.pi, 0.0, image_res[0])
    )
    phis, thetas = np.ravel(phis), np.ravel(thetas)
    return [np.cos(phis) * np.sin(thetas), np.sin(phis) * np.sin(thetas), np.cos(thetas)]

def get_spherical_rays(image_res):
    phis, thetas = dr.meshgrid(
        dr.linspace(mi.Float, 0.0, dr.two_pi, image_res[1]),
        dr.linspace(mi.Float, dr.pi, 0.0, image_res[0])
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
        return mi.TensorXf(dr.ravel(plugin.eval(si)), (*resolution, 3))

