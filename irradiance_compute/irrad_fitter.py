import numpy as np
import matplotlib.pyplot as plt

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

datas = mi.tensor_io.read("datasets/sampling_data.bin")

def fit_irradiance(dataset: np.ndarray, features: list[tuple[int, int]]):

    turbs = np.linspace(1, 10, dataset.shape[0])
    etas  = np.linspace(0, np.pi/2 - 0.01, dataset.shape[1])
    X, Y = np.meshgrid(etas, turbs, copy=False)
    Z = dataset

    X = X.flatten()
    Y = Y.flatten()

    A = np.array([(X + (0.001 if i < 0 else 0))**i * (Y + (0.001 if j < 0 else 0))**j for i, j in features]).T
    B = Z.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B)
    return A, coeff, r

def viz_fitt_sep(sun_original, sun_recovered, sky_original, sky_recovered):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)

    irrad_min = min(sun_original.min(), sun_recovered.min(), sky_original.min(), sky_recovered.min())
    irrad_max = max(sun_original.max(), sun_recovered.max(), sky_original.max(), sky_recovered.max())

    def plot_collumn(sun_data, sky_data, col_title, col_idx):
        # Plot sky irradiance
        im1 = axs[0][col_idx].imshow(sky_data, aspect='auto', vmin=irrad_min, vmax=irrad_max)
        axs[0][col_idx].set_title(f"{col_title} sky irradiance")
        axs[0][col_idx].invert_yaxis()
        axs[0][col_idx].set_yticks(np.linspace(0, sky_data.shape[0] - 1, 10))
        axs[0][col_idx].set_yticklabels(np.linspace(1, 10, 10))

        # Plot sun irradiance
        im2 = axs[1][col_idx].imshow(sun_data, aspect='auto', vmin=irrad_min, vmax=irrad_max)
        axs[1][col_idx].set_title(f"{col_title} sun irradiance")
        axs[1][col_idx].invert_yaxis()
        axs[1][col_idx].set_yticks(np.linspace(0, sun_data.shape[0] - 1, 10))
        axs[1][col_idx].set_yticklabels(np.linspace(1, 10, 10))

        # Plot sampling weights
        diff = sky_data / (sky_data + sun_data)
        im3 = axs[2][col_idx].imshow(diff, aspect='auto', cmap='bwr')
        axs[2][col_idx].set_xlabel("Sun elevation (degrees)")
        axs[2][col_idx].set_title(f"{col_title} sky sampling weight")
        axs[2][col_idx].invert_yaxis()
        axs[2][col_idx].set_yticks(np.linspace(0, diff.shape[0] - 1, 10))
        axs[2][col_idx].set_yticklabels(np.linspace(1, 10, 10))

        if col_idx == 0:
            axs[0][col_idx].set_ylabel("Turbidity")
            axs[1][col_idx].set_ylabel("Turbidity")
            axs[2][col_idx].set_ylabel("Turbidity")

        return im1, im2, im3

    im1, im2, im3 = plot_collumn(sun_original, sky_original, "Original", 0)
    im1, im2, im3 = plot_collumn(sun_recovered, sky_recovered, "Recovered", 1)
    # Add a single colorbar for the first two subplots
    cbar1 = fig.colorbar(im1, ax=axs[:2], orientation='vertical', fraction=0.05, pad=0.1)
    cbar1.set_label("Irradiance")

    # Add a separate colorbar for the third subplot
    cbar2 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.05, pad=0.1)
    cbar2.set_label("Sampling Weight")

    plt.show()

def viz_fitt_total(original, recovered):
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    irrad_min = min(original.min(), recovered.min())
    irrad_max = max(original.max(), recovered.max())

    im1 = axs[0].imshow(original, aspect='auto', vmin=irrad_min, vmax=irrad_max)
    axs[0].set_title(f"Original sampling weight")
    axs[0].invert_yaxis()
    axs[0].set_yticks(np.linspace(0, original.shape[0] - 1, 10))
    axs[0].set_yticklabels(np.linspace(1, 10, 10))

    # Plot sun irradiance
    im2 = axs[1].imshow(recovered, aspect='auto', vmin=irrad_min, vmax=irrad_max)
    axs[1].set_title(f"Recovered sampling weight")
    axs[1].invert_yaxis()
    axs[1].set_yticks(np.linspace(0, recovered.shape[0] - 1, 10))
    axs[1].set_yticklabels(np.linspace(1, 10, 10))

    # Plot sampling weights
    diff = original - recovered
    im3 = axs[2].imshow(diff, aspect='auto', cmap='bwr')
    axs[2].set_xlabel("Sun elevation (degrees)")
    axs[2].set_title(f"Difference in sampling weight")
    axs[2].invert_yaxis()
    axs[2].set_yticks(np.linspace(0, diff.shape[0] - 1, 10))
    axs[2].set_yticklabels(np.linspace(1, 10, 10))

    axs[0].set_ylabel("Turbidity")
    axs[1].set_ylabel("Turbidity")
    axs[2].set_ylabel("Turbidity")

    # Add a single colorbar for the first two subplots
    cbar1 = fig.colorbar(im1, ax=axs[:2], orientation='vertical', fraction=0.05, pad=0.1)
    cbar1.set_label("Irradiance")

    # Add a separate colorbar for the third subplot
    cbar2 = fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.05, pad=0.1)
    cbar2.set_label("Difference")

    plt.show()


if True:
    base_features = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
    features = [(i, j) for i in range(6) for j in range(6)]

    P_sun, coeff_sun, r_sun = fit_irradiance(datas["sun_irradiance"], features)
    P_sky, coeff_sky, r_sky = fit_irradiance(datas["sky_irradiance"], features)

    nb_add_features = 2
    sky_important_idx = np.argsort(np.abs(coeff_sky))
    sun_important_idx = np.argsort(np.abs(coeff_sun))

    # print("Important sky features:", np.array(features)[sky_important_idx])
    # print("Important sun features:", np.array(features)[sun_important_idx])

    if False:
        def select_important_features(base_features, nb_add_features, important_features_idx, all_features):
            selected = base_features.copy()
            nb_added = 0
            for idx in important_features_idx:
                f = tuple(all_features[idx])
                if f not in selected:
                    selected.append(f)
                    nb_added += 1
                
                if nb_added >= nb_add_features:
                    break

            return selected

        sky_features = select_important_features(base_features, nb_add_features, sky_important_idx, features)
        sun_features = select_important_features(base_features, nb_add_features, sun_important_idx, features)
    else:
        sun_features = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0)]#, (0, 2), (0, 6)]
        sky_features = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 5)]#, (0, 2), (2, 5)]

    P_sun, coeff_sun, r_sun = fit_irradiance(datas["sun_irradiance"], sun_features)
    P_sky, coeff_sky, r_sky = fit_irradiance(datas["sky_irradiance"], sky_features)

    print(f"Sun fitting residual: {r_sun}", f"Coefficients: {coeff_sun}", sep="\n")
    print(f"Sky fitting residual: {r_sky}", f"Coefficients: {coeff_sky}", sep="\n")

    recovered_sun = (P_sun @ coeff_sun).reshape(datas["sun_irradiance"].shape).clip(min=0)
    recovered_sky = (P_sky @ coeff_sky).reshape(datas["sky_irradiance"].shape).clip(min=0)

    viz_fitt_sep(datas["sun_irradiance"], recovered_sun, datas["sky_irradiance"], recovered_sky)
else:
    features = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]

    dataset = datas["sky_irradiance"] / (datas["sky_irradiance"] + datas["sun_irradiance"])
    P_total, coeff_total, r_total = fit_irradiance(dataset, features)

    print(f"For features: {features}:")
    print(f"Total fitting residual: {r_total}", f"Coefficients: {coeff_total}", sep="\n")

    recovered_total = (P_total @ coeff_total).reshape(dataset.shape)
    recovered_total /= recovered_total.max()

    viz_fitt_total(dataset, recovered_total)
