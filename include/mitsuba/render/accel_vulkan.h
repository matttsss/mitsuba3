/*
    accel_vulkan.h -- Vulkan acceleration backend declarations.
*/

#pragma once

#if defined(MI_ENABLE_VULKAN)

#include <mitsuba/render/fwd.h>
#include <drjit/array_traverse.h>

NAMESPACE_BEGIN(mitsuba)

/// Opaque handle owning the native Vulkan objects, see src/render/vulkan_accel.cpp
struct VulkanAccelData;

/// Vectorized GPU ray tracing acceleration via Vulkan (VK_KHR_acceleration_structure + VK_KHR_ray_query)
template <typename Float, typename Spectrum>
struct VulkanAccel {
    MI_IMPORT_TYPES(Shape, ShapePtr)
    VulkanAccel() = default;
    DRJIT_NON_COPYABLE(VulkanAccel)

    ~VulkanAccel() { release(); }

    // --- Lifecycle (bodies in scene_vulkan.inl) ---
    void init(Scene<Float, Spectrum> *scene, const Properties &props);
    void rebuild(Scene<Float, Spectrum> *scene);
    void release();

    static void static_initialization() { }
    static void static_shutdown() { }

    // --- Ray queries (bodies in scene_vulkan.inl) ---
    PreliminaryIntersection3f ray_intersect_preliminary(
        const Scene<Float, Spectrum> *scene, const Ray3f &ray, Mask coherent,
        bool reorder, UInt32 reorder_hint, uint32_t reorder_hint_bits,
        Mask active, UInt32 ray_mask) const;
    ShadowTest<Mask> ray_test(const Scene<Float, Spectrum> *scene,
                              const Ray3f &ray, Mask coherent, Mask active,
                              UInt32 ray_mask,
                              bool skip_null) const;
    static constexpr bool stops_at_first_hit = true;
    /// Vulkan has no brute-force traversal; defer to the accelerated path.
    SurfaceInteraction3f ray_intersect_naive(
        const Scene<Float, Spectrum> *scene, const Ray3f &ray,
        Mask active) const;

    // --- Declarative traversal (scene handle + recovery table) ---
    DRJIT_TRAVERSE(VulkanAccel, accel_handle, geom_shape_table)

    /// Opaque handle owning the Vulkan objects (TLAS/BLAS/buffers)
    VulkanAccelData *accel = nullptr;
    /// Dr.Jit scene id from jit_vulkan_configure_scene(), 0 for empty scenes
    uint32_t scene_index = 0;
    /// Handle variable representing the Vulkan scene for @dr.freeze
    UInt64 accel_handle;
    /// Recovery table indexed by TLAS userID + geometry ID, resolving a hit
    /// into ``pi.shape`` (see scene_vulkan.inl)
    DynamicBuffer<UInt32> geom_shape_table;
    /// Layout of ``geom_shape_table``: (shape id, instance index) pairs when
    /// true, plain shape ids otherwise
    bool has_instances = false;

private:
    /// Trace ``ray``, writing eight result variable indices to ``out``. With
    /// ``shadow``, an occlusion query writes only ``out[0]``.
    void trace(const Ray3f &ray, Mask active, UInt32 ray_mask,
               uint32_t out[8], bool shadow) const;
};

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_VULKAN
