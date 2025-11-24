import drjit as dr
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

t_sunsky = mi.load_dict({
        "type": "timed_sunsky",
        "complex_sun": False,
        "sky_scale": 0.0,
        "sun_scale": 0.0,
})
dr.eval(t_sunsky)

with dr.scoped_set_flag(dr.JitFlag.PrintIR, True):
    sample, weight = t_sunsky.sample_direction(dr.zeros(mi.Interaction3f), mi.Point2f(0.), True)
    dr.eval(sample, weight)
