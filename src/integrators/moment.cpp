#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)


/**!

.. _integrator-moment:

Moment integrator (:monosp:`moment`)
-----------------------------------------------------

.. pluginparameters::

 * - (Nested plugin)
   - :paramtype:`integrator`
   - Sub-integrators (can have more than one) which will be sampled along the AOV integrator. Their
     respective XYZ output will be put into distinct images.

This integrator returns one AOVs recording the second moment of the samples of the nested
integrator.

.. tabs::
    .. code-tab:: xml

        <integrator type="moment">
            <integrator type="path"/>
        </integrator>

    .. code-tab:: python

        'type': 'moment',
        'nested': {
            'type': 'path',
        }

 */

template <typename Float, typename Spectrum>
class MomentIntegrator final : public SamplingIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SamplingIntegrator)
    MI_IMPORT_TYPES(Scene, Sampler, Medium)

    MomentIntegrator(const Properties &props) : Base(props) {
        // Get the nested integrators and their AOVs
        for (auto &prop : props.objects()) {
            Base *integrator = prop.try_get<Base>();
            if (!integrator)
                Throw("Child objects must be of type 'SamplingIntegrator'!");
            std::string name(prop.name());
            std::vector<std::string> aovs = integrator->aov_names();
            for (auto aov_name: aovs)
                m_aov_names.push_back(name + "." + aov_name);
            m_integrators.push_back({ integrator, aovs.size() });

            m_aov_names.push_back(name + ".X");
            m_aov_names.push_back(name + ".Y");
            m_aov_names.push_back(name + ".Z");
        }

        // For every AOV, add a corresponding "m2_" AOV
        size_t aov_count = m_aov_names.size();
        for (size_t i = 0; i < aov_count; i++)
            m_aov_names.push_back("m2_" + m_aov_names[i]);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler * sampler,
                                     const RayDifferential3f &ray,
                                     const Medium *medium,
                                     Float *aovs,
                                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        std::pair<Spectrum, Mask> result { 0.f, false };

        size_t offset = m_aov_names.size() / 2;

        for (size_t i = 0; i < m_integrators.size(); i++) {
            std::pair<Spectrum, Mask> result_sub =
                m_integrators[i].first->sample(scene, sampler, ray, medium, aovs, active);
            aovs += m_integrators[i].second;

            UnpolarizedSpectrum spec_u = unpolarized_spectrum(result_sub.first);

            Color3f xyz;
            if constexpr (is_monochromatic_v<Spectrum>) {
                xyz = spec_u.x();
            } else if constexpr (is_rgb_v<Spectrum>) {
                xyz = srgb_to_xyz(spec_u, active);
            } else {
                static_assert(is_spectral_v<Spectrum>);
                /// Note: this assumes that sensor used sample_rgb_spectrum() to generate 'ray.wavelengths'
                auto pdf = pdf_rgb_spectrum(ray.wavelengths);
                spec_u *= dr::select(pdf != 0.f, dr::rcp(pdf), 0.f);
                xyz = spectrum_to_xyz(spec_u, ray.wavelengths, active);
            }

            *aovs++ = xyz.x(); *aovs++ = xyz.y(); *aovs++ = xyz.z();

            // Write second moment AOVs
            for (size_t j = 0; j < m_integrators[i].second + 3; j++)
                *(aovs - j + offset - 1) = dr::square(*(aovs - j - 1));

            if (i == 0)
                result = result_sub;
        }

        return result;
    }

    std::vector<std::string> aov_names() const override {
        return m_aov_names;
    }

    void traverse(TraversalCallback *cb) override {
        for (size_t i = 0; i < m_integrators.size(); ++i)
            cb->put("integrator_" + std::to_string(i),
                                 m_integrators[i].first.get(),
                                 ParamFlags::Differentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Scene[" << std::endl
            << "  aovs = " << m_aov_names << "," << std::endl
            << "  integrators = [" << std::endl;
        for (size_t i = 0; i < m_integrators.size(); ++i) {
            oss << "    " << string::indent(m_integrators[i].first, 4);
            if (i + 1 < m_integrators.size())
                oss << ",";
            oss << std::endl;
        }
        oss << "  ]"<< std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(MomentIntegrator)
private:
    std::vector<std::string> m_aov_names;
    std::vector<std::pair<ref<Base>, size_t>> m_integrators;

    MI_TRAVERSE_CB(Base, m_integrators)
};

MI_EXPORT_PLUGIN(MomentIntegrator)
NAMESPACE_END(mitsuba)
