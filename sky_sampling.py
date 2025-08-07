import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

class CroppedSphericalDomain(mi.chi2.SphericalDomain):
    """
    Cropped spherical domain that avoids the singularity at the north pole by SIN_OFFSET
    """
    def bounds(self):
        cos_bound = dr.sqrt(1 - dr.square(0.00775))
        return mi.ScalarBoundingBox2f([-dr.pi, -cos_bound], [dr.pi, 1])
    

def test06_sky_sampling(turb, sun_theta):
    phi_sun = -4*dr.pi/5
    sp_sun, cp_sun = dr.sincos(phi_sun)
    st, ct = dr.sincos(sun_theta)

    sky = {
        "type": "sunsky",
        "sun_direction": [cp_sun * st, sp_sun * st, ct],
        "sun_scale": 0.0,
        "turbidity": turb,
        "albedo": 0.5
    }

    sample_func, pdf_func = mi.chi2.EmitterAdapter("sunsky", sky)
    test = mi.chi2.ChiSquareTest(
        domain=CroppedSphericalDomain(),
        pdf_func= pdf_func,
        sample_func= sample_func,
        sample_dim=2,
        sample_count=200_000,
        res=55,
        ires=32
    )

    assert test.run(), "Chi2 test failed"


turb = [2.2, 4.8, 6.0]
sun_theta = [dr.deg2rad(20), dr.deg2rad(50)]

tgmms = mi.tensor_io.read("../mitsuba3/resources/data/sunsky/output/tgmm_tables.bin")['tgmm_tables']

idx = 1
test06_sky_sampling(turb[idx], sun_theta[idx])
