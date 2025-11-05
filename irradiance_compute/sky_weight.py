from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


import drjit as dr
import mitsuba as mi

def update_plugin(plugin_params, is_sun, t, eta):
    sp_sun, cp_sun = 0, 1 # dr.sincos(0.)
    st, ct = dr.sincos(dr.pi/2 - eta)

    if is_sun:
        plugin_params['sun_scale'] = 1.0
        plugin_params['sky_scale'] = 0.0
    else:
        plugin_params['sun_scale'] = 0.0
        plugin_params['sky_scale'] = 1.0

    plugin_params['turbidity'] = t
    plugin_params['sun_direction'] = mi.Vector3f(cp_sun * st, sp_sun * st, ct)
    plugin_params.update()

@dr.freeze
def get_rays(quad_points, weights, cos_cutoff):
    j = 0.5 * dr.pi * (1. - cos_cutoff)
    phi = dr.pi * (quad_points + 1)
    cos_theta = 0.5 * ((1. - cos_cutoff) * quad_points + (1 + cos_cutoff))

    phi, cos_theta = dr.meshgrid(phi, cos_theta)
    w_phi, w_cos_theta = dr.meshgrid(weights, weights)
    sin_phi, cos_phi = dr.sincos(phi)
    sin_theta = dr.safe_sqrt(1 - cos_theta * cos_theta)

    wo = mi.Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)

    return j, wo, w_phi * w_cos_theta

@dr.freeze
def evaluate_radiance(emitter, si):
    wavelengths = [320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720]

    wav_res = dr.zeros(mi.ArrayXf, (11, 1))
    for i, wav in enumerate(wavelengths):
        si.wavelengths = wav
        radiance = emitter.eval(si)
        wav_res[i] = radiance[0]

    return wav_res


@dr.freeze
def sky_integrand(emitter, quad_points, quad_weights):
    j, sky_wo, weights = get_rays(quad_points, quad_weights, mi.Float(0))

    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = -sky_wo

    return j * evaluate_radiance(emitter, si) * weights


@dr.freeze
def sun_integrand(emitter, quad_points, quad_weights, sun_direction, sun_cos_cutoff):
    j, sun_wo, weights = get_rays(quad_points, quad_weights, sun_cos_cutoff)

    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = -mi.Frame3f(sun_direction).to_world(sun_wo)

    return j * evaluate_radiance(emitter, si) * weights


def viz_sky_wav():

    wavelengths = np.arange(11) * 40 + 320

    d65 = mi.load_dict({ 'type': 'd65' })
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wavelengths = mi.Float(wavelengths)
    d65_pdf  = d65.eval(si, True)[0]
    d65_pdf /= dr.sum(d65_pdf)

    tensors = mi.tensor_io.read("datasets/sampling_data.bin")
    sky_irrad = tensors["sky_irradiance"]
    sun_irrad = tensors["sun_irradiance"]
    irrad_pdf = sky_irrad + sun_irrad
    irrad_pdf /= np.sum(irrad_pdf, axis=2, keepdims=True)
    irrad_pdf = mi.TensorXf(irrad_pdf)

    spectrum_pdf = dr.mean(irrad_pdf, axis=(0, 1))

    irrad_pdf_std_dev = np.std(irrad_pdf, axis=(0, 1))
    print(irrad_pdf_std_dev)


    #plt.plot(wavelengths, irrad_pdf_std_dev, label="Std dev on irradiance dataset")
    plt.plot(wavelengths, spectrum_pdf, label="Average irradiance pdf")
    plt.plot(wavelengths, d65_pdf, label="D65 pdf")

    plt.xlabel("Wavelengths")
    plt.ylabel("PDF")
    plt.legend()

    plt.show()


def viz_irradiance(sun_irrad, sky_irrad, etas):

    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # Plot sky irradiance
    im1 = axs[0].imshow(sky_irrad, aspect='auto', vmin=min(sky_irrad.min(), sun_irrad.min()), vmax=max(sky_irrad.max(), sun_irrad.max()))
    axs[0].set_ylabel("Turbidity")
    axs[0].set_title("Irradiance from sky")
    axs[0].invert_yaxis()
    axs[0].set_yticks(np.linspace(0, sky_irrad.shape[0] - 1, 10))
    axs[0].set_yticklabels(np.linspace(1, 10, 10))

    # Plot sun irradiance
    im2 = axs[1].imshow(sun_irrad, aspect='auto', vmin=min(sky_irrad.min(), sun_irrad.min()), vmax=max(sky_irrad.max(), sun_irrad.max()))
    axs[1].set_ylabel("Turbidity")
    axs[1].set_title("Irradiance from sun")
    axs[1].invert_yaxis()
    axs[1].set_yticks(np.linspace(0, sun_irrad.shape[0] - 1, 10))
    axs[1].set_yticklabels(np.linspace(1, 10, 10))

    # Plot sampling weights
    weights = sky_irrad / (sky_irrad + sun_irrad)
    im3 = axs[2].imshow(weights, aspect='auto')
    axs[2].set_xlabel("Sun elevation (degrees)")
    axs[2].set_ylabel("Turbidity")
    axs[2].set_title("Sampling weight for sky")
    axs[2].invert_yaxis()
    axs[2].set_yticks(np.linspace(0, weights.shape[0] - 1, 10))
    axs[2].set_yticklabels(np.linspace(1, 10, 10))

    # Add a single colorbar for the first two subplots
    cbar1 = fig.colorbar(im1, ax=axs[:2], orientation='vertical', fraction=0.05, pad=0.1)
    cbar1.set_label("Irradiance")

    # Add a separate colorbar for the third subplot
    cbar2 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.05, pad=0.1)
    cbar2.set_label("Sampling Weight")

    plt.show()

if __name__ == "__main__":
    mi.set_variant("cuda_ad_spectral")

    sun_aperture = 0.5358
    sun_cos_cutoff = dr.cos(dr.deg2rad(0.5 * sun_aperture))

    # Load the sunsky emitter
    emitter = mi.load_dict({
        "type": "sunsky",
        "complex_sun": True,
        "albedo": 0.5,
        "sun_aperture": sun_aperture,
        "sun_direction": [0, 0, 1],
        "sun_scale": 0.0,
    })
    emitter_params = mi.traverse(emitter)

    res = 200
    quad_points, weights = mi.quad.gauss_legendre(res)

    res = (10, 30)
    turbs = np.linspace(1, 10, res[0])
    min_eta = dr.deg2rad(0.5 * 0.5358) + 1e-2
    etas  = (2 * np.linspace(min_eta, dr.pi/2 - min_eta, res[1]) * dr.inv_pi)**3 * dr.pi/2
    print(etas * 2 * dr.inv_pi)

    plt.plot(etas * 2 * dr.inv_pi)
    plt.show()
    sky_spec_irradiance = np.zeros((*res, 11), dtype=np.float32)
    for i, turb in enumerate(turbs):
        for j, eta in enumerate(etas):
            update_plugin(emitter_params, False, turb, eta)

            integrand = sky_integrand(emitter, quad_points, weights)
            integrand = dr.sum(integrand, axis=1).numpy().squeeze()
            sky_spec_irradiance[i, j] = integrand

    sun_spec_irradiance = np.zeros((*res, 11), dtype=np.float32)
    for i, turb in enumerate(turbs):
        for j, eta in enumerate(etas):
            update_plugin(emitter_params, True, turb, eta)

            sun_dir = emitter_params['sun_direction']

            integrand = sun_integrand(emitter, quad_points, weights, sun_dir, sun_cos_cutoff)
            sun_spec_irradiance[i, j] = dr.sum(integrand, axis=1).numpy().squeeze()


    sun_lum = np.zeros(np.prod(res), dtype=np.float32)
    sky_lum = np.zeros(np.prod(res), dtype=np.float32)
    for i in range(11):

        irrad = np.ravel(sun_spec_irradiance[::, ::, i])
        sun_lum += mi.luminance(mi.Spectrum(0) + mi.Float(irrad), mi.Spectrum(320 + i * 40)).numpy()
        irrad = np.ravel(sky_spec_irradiance[::, ::, i])
        sky_lum += mi.luminance(mi.Spectrum(0) + mi.Float(irrad), mi.Spectrum(320 + i * 40)).numpy()

    sun_lum /= 11
    sky_lum /= 11
    sun_lum = sun_lum.reshape(res)
    sky_lum = sky_lum.reshape(res)

    mi.tensor_io.write("datasets/luminance_data.bin",
        sky_lum=sky_lum,
        sun_lum=sun_lum
    )

    viz_irradiance(sun_lum, sky_lum, dr.rad2deg(etas))