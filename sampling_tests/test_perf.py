import numpy as np
import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


sampling_type = "tgmm"
seed = mi.UInt32(0)
rng = dr.rng(seed)
max_spp = 200
spp = dr.arange(mi.UInt32, max_spp)

plugin: mi.Emitter = mi.load_dict({
        "type": "timed_sunsky",
        "sun_scale": 0
    })


@dr.syntax
def estimate(plugin: mi.Emitter, rng, spp: mi.UInt32):
    res = mi.Spectrum(0)
    it = dr.zeros(mi.Interaction3f)
    
    i = mi.UInt32(0)
    active = i < spp
    while active:
        sample1 = rng.uniform(mi.Float, len(spp))
        sample2 = rng.uniform(mi.Float, len(spp))

        _, spec = plugin.sample_direction(it, mi.Point2f(sample1, sample2), active)
        
        res += spec
        i += 1
        active = i < spp
    
    return res / mi.Float(spp)

irradiance = estimate(plugin, rng, spp)
irradiance = mi.luminance(irradiance)

data = mi.tensor_io.read("sampling_tests/out/irrad.bin")
data |= {sampling_type: irradiance}
mi.tensor_io.write("sampling_tests/out/irrad.bin", **data)

# avg = np.array(irradiance)
# avg = np.mean(avg[600:])
avg = 0.121143416

rae_uniff = np.abs(data["unif"] - avg) / avg
rae_tgmm = np.abs(data["tgmm"] - avg) / avg

#plt.plot(spp, irradiance)
plt.plot(spp, rae_uniff, label="unif")
plt.plot(spp, rae_tgmm, label="tgmm")
plt.xlabel("Number of samples")
plt.ylabel("Relative Absolute Error")
plt.title("RAE as a function of sample count")
plt.legend()
#plt.hlines(avg, 0, max_spp, colors=["g"])
plt.show()
