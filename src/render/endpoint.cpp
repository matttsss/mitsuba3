#include <mutex>

#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/endpoint.h>
#include <mitsuba/render/medium.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT Endpoint<Float, Spectrum>::Endpoint(const Properties &props)
    : JitObject<Endpoint>(props.id(), ObjectType::Unknown) {
    m_to_world = props.get<ScalarAffineTransform4f>("to_world", ScalarAffineTransform4f());
    dr::make_opaque(m_to_world);

    for (auto &prop : props.objects()) {
        if (Medium *medium = prop.try_get<Medium>()) {
            if (m_medium)
                Throw("Only a single medium can be specified per endpoint (e.g. per emitter or sensor)");
            set_medium(medium);
        }
    }
}

MI_VARIANT Endpoint<Float, Spectrum>::Endpoint(const Properties &props, ObjectType type)
    : JitObject<Endpoint>(props.id(), type) {
    m_to_world = props.get<ScalarAffineTransform4f>("to_world", ScalarAffineTransform4f());
    dr::make_opaque(m_to_world);

    for (auto &prop : props.objects()) {
        if (Medium *medium = prop.try_get<Medium>()) {
            if (m_medium)
                Throw("Only a single medium can be specified per endpoint (e.g. per emitter or sensor)");
            set_medium(medium);
        }
    }
}

MI_VARIANT Endpoint<Float, Spectrum>::~Endpoint() { }

MI_VARIANT void Endpoint<Float, Spectrum>::set_scene(const Scene *) { }

static std::mutex set_dependency_lock;

MI_VARIANT void Endpoint<Float, Spectrum>::set_shape(Shape *shape) {
    std::unique_lock<std::mutex> guard(set_dependency_lock);
    if (m_shape)
        Throw("An endpoint can be only be attached to a single shape.");

    m_shape = shape;
}

MI_VARIANT void Endpoint<Float, Spectrum>::set_medium(Medium *medium) {
    std::unique_lock<std::mutex> guard(set_dependency_lock);
    if (m_medium)
        Throw("An endpoint can be only be attached to a single medium.");

    m_medium = medium;
}

MI_VARIANT std::pair<typename Endpoint<Float, Spectrum>::Ray3f, Spectrum>
Endpoint<Float, Spectrum>::sample_ray(Float /*time*/,
                                      Float /*sample1*/,
                                      const Point2f & /*sample2*/,
                                      const Point2f & /*sample3*/,
                                      Mask /*active*/) const {
    NotImplementedError("sample_ray");
}

MI_VARIANT std::pair<typename Endpoint<Float, Spectrum>::DirectionSample3f, Spectrum>
Endpoint<Float, Spectrum>::sample_direction(const Interaction3f & /*it*/,
                                            const Point2f & /*sample*/,
                                            Mask /*active*/) const {
    NotImplementedError("sample_direction");
}

MI_VARIANT
std::pair<typename Endpoint<Float, Spectrum>::PositionSample3f, Float>
Endpoint<Float, Spectrum>::sample_position(Float /*time*/,
                                           const Point2f &/*sample*/,
                                           Mask /*active*/) const {
    NotImplementedError("sample_position");
}

MI_VARIANT std::pair<typename Endpoint<Float, Spectrum>::Wavelength, Spectrum>
Endpoint<Float, Spectrum>::sample_wavelengths(const SurfaceInteraction3f & /*si*/,
                                              Float /*sample*/,
                                              Mask /*active*/) const {
    NotImplementedError("sample_wavelengths");
}

MI_VARIANT Float Endpoint<Float, Spectrum>::pdf_direction(const Interaction3f & /*it*/,
                                                           const DirectionSample3f & /*ds*/,
                                                           Mask /*active*/) const {
    NotImplementedError("pdf_direction");
}

MI_VARIANT Spectrum
Endpoint<Float, Spectrum>::eval_direction(const Interaction3f & /*it*/,
                                          const DirectionSample3f & /*ds*/,
                                          Mask /*active*/) const {
    NotImplementedError("eval_direction");
}

MI_VARIANT Float Endpoint<Float, Spectrum>::pdf_position(
    const PositionSample3f & /*ps*/, Mask /*active*/) const {
    NotImplementedError("pdf_position");
}

MI_VARIANT Spectrum Endpoint<Float, Spectrum>::pdf_wavelengths(
    const Spectrum & /*wavelengths*/, Mask /*active*/) const {
    NotImplementedError("pdf_wavelengths");
}

MI_VARIANT Spectrum Endpoint<Float, Spectrum>::eval(
    const SurfaceInteraction3f & /*si*/, Mask /*active*/) const {
    NotImplementedError("eval");
}

MI_VARIANT void Endpoint<Float, Spectrum>::traverse(TraversalCallback *cb) {
    if (m_medium)
        cb->put("medium", m_medium, ParamFlags::Differentiable);
}

MI_VARIANT void Endpoint<Float, Spectrum>::parameters_changed(const std::vector<std::string> &keys) {
    if (keys.empty() || string::contains(keys, "to_world")) {
        m_to_world = m_to_world.value().update();
        dr::make_opaque(m_to_world);
    }
}

MI_IMPLEMENT_TRAVERSE_CB(Endpoint, Object)
MI_INSTANTIATE_CLASS(Endpoint)
NAMESPACE_END(mitsuba)
