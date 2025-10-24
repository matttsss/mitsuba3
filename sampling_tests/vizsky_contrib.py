from __future__ import annotations
import numpy as np

import drjit as dr
import mitsuba as mi

def load_tensors(file_path):
    tensors = mi.tensor_io.read(file_path)

    sky_rad = tensors[f"sky_rad_{'rgb' if mi.is_rgb else 'spec'}"]
    sky_params = tensors[f"sky_params_{'rgb' if mi.is_rgb else 'spec'}"]

    return mi.TensorXf(sky_rad), mi.TensorXf(sky_params)

@dr.freeze
def interpolate_dataset(dataset, albedo, turbidity, eta):
    dataset = dr.take_interp(dataset, turbidity - 1)
    dataset = dr.take_interp(dataset, albedo)

    eta = (2. * eta * dr.inv_pi) ** (1/3)

    res = 0
    coefs = 1, 5, 10, 10, 5, 1
    for i in range(6):
        res += dataset[i] * (eta**i) * ((1 - eta)**(5 - i)) * coefs[i]

    return res

def eval(sky_params, cos_theta, gamma):
    cos_gamma = dr.cos(gamma)
    cos_gamma_2 = cos_gamma * cos_gamma

    c1 = 1 + sky_params[..., 0, dr.newaxis] * dr.exp(sky_params[..., 1, dr.newaxis] / (cos_theta + 0.01))
    chi = (1 + cos_gamma_2) / ((1 + sky_params[..., 8, dr.newaxis]**2 - 2 * sky_params[..., 8, dr.newaxis] * cos_gamma)**1.5)
    c2 = sky_params[..., 2, dr.newaxis] + sky_params[..., 3, dr.newaxis] * dr.exp(sky_params[..., 4, dr.newaxis] * gamma) + \
         sky_params[..., 5, dr.newaxis] * cos_gamma_2 + sky_params[..., 6, dr.newaxis] * chi + sky_params[..., 7, dr.newaxis] * dr.safe_sqrt(cos_theta)

    return c1 * c2

@dr.freeze()
def f1(sky_params, cos_theta, gamma):
    return sky_params[..., 2, dr.newaxis]

@dr.freeze()
def f2(sky_params, cos_theta, gamma):
    return sky_params[..., 3, dr.newaxis] * dr.exp(sky_params[..., 4, dr.newaxis] * gamma)

@dr.freeze()
def f3(sky_params, cos_theta, gamma):
    cos_gamma = dr.cos(gamma)
    cos_gamma_2 = cos_gamma * cos_gamma
    return sky_params[..., 5, dr.newaxis] * cos_gamma_2

@dr.freeze()
def f4(sky_params, cos_theta, gamma):
    cos_gamma = dr.cos(gamma)
    cos_gamma_2 = cos_gamma * cos_gamma
    chi = (1 + cos_gamma_2) / ((1 + sky_params[..., 8, dr.newaxis]**2 - 2 * sky_params[..., 8, dr.newaxis] * cos_gamma)**1.5)
    return sky_params[..., 6, dr.newaxis] * chi

@dr.freeze()
def f5(sky_params, cos_theta, gamma):
    return sky_params[..., 7, dr.newaxis] * dr.safe_sqrt(cos_theta)

@dr.freeze()
def f_extra(sky_params, cos_theta, gamma):
    return sky_params[..., 0, dr.newaxis] * dr.exp(sky_params[..., 1, dr.newaxis] / (cos_theta + 0.01))

def eval_luminance(func, sky_rad, sky_params, cos_theta, gamma):
    res = sky_rad[..., dr.newaxis] * func(sky_params, cos_theta, gamma) & (cos_theta >= 0)

    nb_channels = sky_rad.shape[0]

    if mi.is_rgb:
        lum = mi.luminance(mi.Color3f(res)) * mi.MI_CIE_D65_NORMALIZATION
    else:
        lum = 0
        for i in range(nb_channels):
            lum += mi.luminance(mi.Spectrum(res[i]), mi.Spectrum(320 + i * 40))
        lum /= nb_channels

        lum *= mi.MI_CIE_D65_NORMALIZATION

    return lum


def compute_parts(funcs, extra_funcs, sky_rad, sky_params, cos_theta, gamma):
    from itertools import product

    parts = []
    for f, extra_f in product(funcs, [lambda *args: 1] + extra_funcs):
        partial_pdf = lambda sky_params, cos_theta, gamma: extra_f(sky_params, cos_theta, gamma) * f(sky_params, cos_theta, gamma)
        res = eval_luminance(partial_pdf, sky_rad, sky_params, cos_theta, gamma)
        res = mi.TensorXf(res, shape=image_res)
        parts.append(res)

    return parts

def make_avg(sky_rad_dataset, sky_params_dataset, view_dirs, elevation_res=20, t_res=20, alb_res=6):
    etas = np.linspace(0, np.pi/2, elevation_res)
    turbs = np.linspace(1, 10, t_res)
    alb = np.linspace(0, 1, alb_res)


    res = 0
    from itertools import product
    for (eta, turb, alb) in product(etas, turbs, alb):
        
        se, ce = dr.sincos(eta)
        sun_dir = mi.Vector3f(0.0, se, ce)
        cos_theta = view_dirs.z
        gamma = dr.unit_angle(sun_dir, view_dirs)

        sky_rad = interpolate_dataset(sky_rad_dataset, alb, turb, eta)
        sky_params = interpolate_dataset(sky_params_dataset, alb, turb, eta)

        res += eval_luminance(eval, sky_rad, sky_params, cos_theta, gamma)

    res = mi.TensorXf(res, shape=image_res)
    return res / (elevation_res * t_res * alb_res)


if __name__ == "__main__":

    mi.set_variant('cuda_ad_rgb')

    from helpers import get_spherical_rays

    image_res = (64, 128)
    turb, albedo, sun_eta = 3.0, 0.3, dr.pi/4
    sun_dir = mi.Vector3f(0.0, dr.sin(sun_eta), dr.cos(sun_eta))

    sky_rad, sky_params = load_tensors('../mitsuba3/resources/data/sunsky/output/sunsky_datasets.bin')
    view_dirs = get_spherical_rays(image_res)

    dr.print("Sky radiance shape: %s, Sky parameters shape: %s" % (sky_rad.shape, sky_params.shape))
    dr.eval(sky_rad, sky_params, view_dirs)

    if True:
        avg_lum = make_avg(sky_rad, sky_params, view_dirs)
        mi.util.write_bitmap("sampling_tests/out/avg_partial_lum.exr", avg_lum)
    else:

        cos_theta = view_dirs.z
        gamma = dr.unit_angle(sun_dir, view_dirs)

        sky_rad = interpolate_dataset(sky_rad, albedo, turb, sun_eta)
        sky_params = interpolate_dataset(sky_params, albedo, turb, sun_eta)

        funcs = [f1, f2, f3, f4, f5]
        parts = compute_parts(funcs, [f_extra], sky_rad, sky_params, cos_theta, gamma)

        for i, part in enumerate(parts):
            mi.util.write_bitmap(f"sampling_tests/parts/part_{i+1}.exr", part)