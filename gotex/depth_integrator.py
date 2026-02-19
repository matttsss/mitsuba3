import drjit as dr
import mitsuba as mi

class InfDepthIntegrator(mi.SamplingIntegrator):
    def __init__(self, props):
        mi.SamplingIntegrator.__init__(self, props)

    def sample(self, 
               scene: mi.Scene, sampler, ray, medium, aovs, active = True):
        pi = scene.ray_intersect_preliminary(ray, coherent=True, active=active)
        return dr.select(pi.is_valid(), pi.t, dr.inf), pi.is_valid(), []
    
mi.register_integrator("infdepth", lambda props: InfDepthIntegrator(props))