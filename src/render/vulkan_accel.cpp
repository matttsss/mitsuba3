/*
    vulkan_accel.cpp -- Vulkan acceleration structure builder.
*/

#include "vulkan/accel.h"

#if defined(MI_ENABLE_VULKAN)

#include <mitsuba/core/logger.h>
#include <mitsuba/core/util.h>
#include <drjit-core/vulkan.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

// ============================================================================
//  Minimal self-contained Vulkan declarations for VK_KHR_acceleration_structure
// ============================================================================

#if defined(_WIN32)
#  define MI_VKAPI_PTR __stdcall
#else
#  define MI_VKAPI_PTR
#endif

namespace {

using VkFlags         = uint32_t;
using VkBool32        = uint32_t;
using VkDeviceSize    = uint64_t;
using VkDeviceAddress = uint64_t;

using VkDevice                   = void *;
using VkQueue                    = void *;
using VkCommandPool              = void *;
using VkCommandBuffer            = void *;
using VkFence                    = void *;
using VkBuffer                   = void *;
using VkQueryPool                = void *;
using VkAccelerationStructureKHR = void *;

constexpr uint32_t VK_TRUE_VAL  = 1U;
constexpr uint32_t VK_FALSE_VAL = 0U;

// VkStructureType values
constexpr uint32_t VK_STRUCTURE_TYPE_SUBMIT_INFO                                   = 4;
constexpr uint32_t VK_STRUCTURE_TYPE_FENCE_CREATE_INFO                             = 8;
constexpr uint32_t VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO                        = 11;
constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO                      = 39;
constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO                  = 40;
constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO                     = 42;
constexpr uint32_t VK_STRUCTURE_TYPE_MEMORY_BARRIER                                = 46;
constexpr uint32_t VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR = 1000150000;
constexpr uint32_t VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR = 1000150002;
constexpr uint32_t VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR = 1000150003;
constexpr uint32_t VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR = 1000150004;
constexpr uint32_t VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR = 1000150005;
constexpr uint32_t VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR           = 1000150006;
constexpr uint32_t VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR          = 1000150010;
constexpr uint32_t VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR        = 1000150017;
constexpr uint32_t VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR   = 1000150020;

// Enumerations & bitmasks
constexpr uint32_t VK_COMMAND_POOL_CREATE_TRANSIENT_BIT            = 0x00000001;
constexpr uint32_t VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002;
constexpr uint32_t VK_COMMAND_BUFFER_LEVEL_PRIMARY                 = 0;
constexpr uint32_t VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT     = 0x00000001;

constexpr uint32_t VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT                   = 0x00000800;
constexpr uint32_t VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR = 0x02000000;

constexpr uint32_t VK_ACCESS_SHADER_READ_BIT                        = 0x00000020;
constexpr uint32_t VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR    = 0x00200000;
constexpr uint32_t VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR   = 0x00400000;

constexpr uint32_t VK_FORMAT_R32G32B32_SFLOAT = 106;
constexpr uint32_t VK_INDEX_TYPE_UINT32       = 1;

constexpr uint32_t VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR    = 0;
constexpr uint32_t VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR = 1;

constexpr uint32_t VK_GEOMETRY_TYPE_TRIANGLES_KHR = 0;
constexpr uint32_t VK_GEOMETRY_TYPE_AABBS_KHR     = 1;
constexpr uint32_t VK_GEOMETRY_TYPE_INSTANCES_KHR = 2;

constexpr uint32_t VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR  = 0;
constexpr uint32_t VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR = 1;

constexpr uint32_t VK_GEOMETRY_OPAQUE_BIT_KHR = 0x00000001;

constexpr uint32_t VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR        = 0x00000001;
constexpr uint32_t VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR     = 0x00000002;

constexpr uint32_t VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR  = 0x00000002;
constexpr uint32_t VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR = 0x00000004;

constexpr uint32_t VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR = 1000150000;
constexpr uint32_t VK_QUERY_RESULT_64_BIT   = 0x00000001;
constexpr uint32_t VK_QUERY_RESULT_WAIT_BIT = 0x00000002;

constexpr uint32_t VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR = 1;

struct VkCommandPoolCreateInfo {
    uint32_t sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
};

struct VkCommandBufferAllocateInfo {
    uint32_t sType;
    const void *pNext;
    VkCommandPool commandPool;
    uint32_t level;
    uint32_t commandBufferCount;
};

struct VkCommandBufferBeginInfo {
    uint32_t sType;
    const void *pNext;
    VkFlags flags;
    const void *pInheritanceInfo;
};

struct VkSubmitInfo {
    uint32_t sType;
    const void *pNext;
    uint32_t waitSemaphoreCount;
    const void *pWaitSemaphores;
    const VkFlags *pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer *pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void *pSignalSemaphores;
};

struct VkFenceCreateInfo {
    uint32_t sType;
    const void *pNext;
    VkFlags flags;
};

struct VkMemoryBarrier {
    uint32_t sType;
    const void *pNext;
    VkFlags srcAccessMask;
    VkFlags dstAccessMask;
};

struct VkQueryPoolCreateInfo {
    uint32_t sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queryType;
    uint32_t queryCount;
    VkFlags pipelineStatistics;
};

union VkDeviceOrHostAddressConstKHR {
    VkDeviceAddress deviceAddress;
    const void *hostAddress;
};

union VkDeviceOrHostAddressKHR {
    VkDeviceAddress deviceAddress;
    void *hostAddress;
};

struct VkAccelerationStructureBuildRangeInfoKHR {
    uint32_t primitiveCount;
    uint32_t primitiveOffset;
    uint32_t firstVertex;
    uint32_t transformOffset;
};

struct VkAccelerationStructureGeometryTrianglesDataKHR {
    uint32_t sType;
    const void *pNext;
    uint32_t vertexFormat;
    VkDeviceOrHostAddressConstKHR vertexData;
    VkDeviceSize vertexStride;
    uint32_t maxVertex;
    uint32_t indexType;
    VkDeviceOrHostAddressConstKHR indexData;
    VkDeviceOrHostAddressConstKHR transformData;
};

struct VkAccelerationStructureGeometryAabbsDataKHR {
    uint32_t sType;
    const void *pNext;
    VkDeviceOrHostAddressConstKHR data;
    VkDeviceSize stride;
};

struct VkAccelerationStructureGeometryInstancesDataKHR {
    uint32_t sType;
    const void *pNext;
    VkBool32 arrayOfPointers;
    VkDeviceOrHostAddressConstKHR data;
};

union VkAccelerationStructureGeometryDataKHR {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles;
    VkAccelerationStructureGeometryAabbsDataKHR aabbs;
    VkAccelerationStructureGeometryInstancesDataKHR instances;
};

struct VkAccelerationStructureGeometryKHR {
    uint32_t sType;
    const void *pNext;
    uint32_t geometryType;
    VkAccelerationStructureGeometryDataKHR geometry;
    VkFlags flags;
};

struct VkAccelerationStructureBuildGeometryInfoKHR {
    uint32_t sType;
    const void *pNext;
    uint32_t type;
    VkFlags flags;
    uint32_t mode;
    VkAccelerationStructureKHR srcAccelerationStructure;
    VkAccelerationStructureKHR dstAccelerationStructure;
    uint32_t geometryCount;
    const VkAccelerationStructureGeometryKHR *pGeometries;
    const VkAccelerationStructureGeometryKHR *const *ppGeometries;
    VkDeviceOrHostAddressKHR scratchData;
};

struct VkAccelerationStructureCreateInfoKHR {
    uint32_t sType;
    const void *pNext;
    VkFlags createFlags;
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
    uint32_t type;
    VkDeviceAddress deviceAddress;
};

struct VkAccelerationStructureDeviceAddressInfoKHR {
    uint32_t sType;
    const void *pNext;
    VkAccelerationStructureKHR accelerationStructure;
};

struct VkAccelerationStructureBuildSizesInfoKHR {
    uint32_t sType;
    void *pNext;
    VkDeviceSize accelerationStructureSize;
    VkDeviceSize updateScratchSize;
    VkDeviceSize buildScratchSize;
};

struct VkCopyAccelerationStructureInfoKHR {
    uint32_t sType;
    const void *pNext;
    VkAccelerationStructureKHR src;
    VkAccelerationStructureKHR dst;
    uint32_t mode;
};

struct VkTransformMatrixKHR {
    float matrix[3][4];
};

struct VkAccelerationStructureInstanceKHR {
    VkTransformMatrixKHR transform;
    uint32_t instanceCustomIndex : 24;
    uint32_t mask : 8;
    uint32_t instanceShaderBindingTableRecordOffset : 24;
    uint32_t flags : 8;
    uint64_t accelerationStructureReference;
};

static_assert(sizeof(VkAccelerationStructureInstanceKHR) == 64,
              "VkAccelerationStructureInstanceKHR must be 64 bytes");

using PFN_vkCreateCommandPool = int32_t (MI_VKAPI_PTR *)(VkDevice, const VkCommandPoolCreateInfo *, const void *, VkCommandPool *);
using PFN_vkDestroyCommandPool = void (MI_VKAPI_PTR *)(VkDevice, VkCommandPool, const void *);
using PFN_vkAllocateCommandBuffers = int32_t (MI_VKAPI_PTR *)(VkDevice, const VkCommandBufferAllocateInfo *, VkCommandBuffer *);
using PFN_vkFreeCommandBuffers = void (MI_VKAPI_PTR *)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer *);
using PFN_vkBeginCommandBuffer = int32_t (MI_VKAPI_PTR *)(VkCommandBuffer, const VkCommandBufferBeginInfo *);
using PFN_vkEndCommandBuffer = int32_t (MI_VKAPI_PTR *)(VkCommandBuffer);
using PFN_vkResetCommandBuffer = int32_t (MI_VKAPI_PTR *)(VkCommandBuffer, VkFlags);
using PFN_vkQueueSubmit = int32_t (MI_VKAPI_PTR *)(VkQueue, uint32_t, const VkSubmitInfo *, VkFence);
using PFN_vkCreateFence = int32_t (MI_VKAPI_PTR *)(VkDevice, const VkFenceCreateInfo *, const void *, VkFence *);
using PFN_vkDestroyFence = void (MI_VKAPI_PTR *)(VkDevice, VkFence, const void *);
using PFN_vkWaitForFences = int32_t (MI_VKAPI_PTR *)(VkDevice, uint32_t, const VkFence *, VkBool32, uint64_t);
using PFN_vkResetFences = int32_t (MI_VKAPI_PTR *)(VkDevice, uint32_t, const VkFence *);
using PFN_vkCmdPipelineBarrier = void (MI_VKAPI_PTR *)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, uint32_t, const VkMemoryBarrier *, uint32_t, const void *, uint32_t, const void *);
using PFN_vkCreateQueryPool = int32_t (MI_VKAPI_PTR *)(VkDevice, const VkQueryPoolCreateInfo *, const void *, VkQueryPool *);
using PFN_vkDestroyQueryPool = void (MI_VKAPI_PTR *)(VkDevice, VkQueryPool, const void *);
using PFN_vkGetQueryPoolResults = int32_t (MI_VKAPI_PTR *)(VkDevice, VkQueryPool, uint32_t, uint32_t, size_t, void *, VkDeviceSize, VkFlags);
using PFN_vkCmdResetQueryPool = void (MI_VKAPI_PTR *)(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t);
using PFN_vkCreateAccelerationStructureKHR = int32_t (MI_VKAPI_PTR *)(VkDevice, const VkAccelerationStructureCreateInfoKHR *, const void *, VkAccelerationStructureKHR *);
using PFN_vkDestroyAccelerationStructureKHR = void (MI_VKAPI_PTR *)(VkDevice, VkAccelerationStructureKHR, const void *);
using PFN_vkGetAccelerationStructureBuildSizesKHR = void (MI_VKAPI_PTR *)(VkDevice, uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR *, const uint32_t *, VkAccelerationStructureBuildSizesInfoKHR *);
using PFN_vkGetAccelerationStructureDeviceAddressKHR = VkDeviceAddress (MI_VKAPI_PTR *)(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR *);
using PFN_vkCmdBuildAccelerationStructuresKHR = void (MI_VKAPI_PTR *)(VkCommandBuffer, uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR *, const VkAccelerationStructureBuildRangeInfoKHR *const *);
using PFN_vkCmdWriteAccelerationStructuresPropertiesKHR = void (MI_VKAPI_PTR *)(VkCommandBuffer, uint32_t, const VkAccelerationStructureKHR *, uint32_t, VkQueryPool, uint32_t);
using PFN_vkCmdCopyAccelerationStructureKHR = void (MI_VKAPI_PTR *)(VkCommandBuffer, const VkCopyAccelerationStructureInfoKHR *);

struct VulkanFunctions {
    PFN_vkCreateCommandPool CreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool DestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers FreeCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
    PFN_vkResetCommandBuffer ResetCommandBuffer = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkCreateFence CreateFence = nullptr;
    PFN_vkDestroyFence DestroyFence = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
    PFN_vkResetFences ResetFences = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCreateQueryPool CreateQueryPool = nullptr;
    PFN_vkDestroyQueryPool DestroyQueryPool = nullptr;
    PFN_vkGetQueryPoolResults GetQueryPoolResults = nullptr;
    PFN_vkCmdResetQueryPool CmdResetQueryPool = nullptr;
    PFN_vkCreateAccelerationStructureKHR CreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR DestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR GetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR GetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR CmdWriteAccelerationStructuresPropertiesKHR = nullptr;
    PFN_vkCmdCopyAccelerationStructureKHR CmdCopyAccelerationStructureKHR = nullptr;

    void resolve() {
        #define MI_VK_RESOLVE(name) \
            name = reinterpret_cast<PFN_vk##name>(jit_vulkan_lookup("vk" #name))
        MI_VK_RESOLVE(CreateCommandPool);
        MI_VK_RESOLVE(DestroyCommandPool);
        MI_VK_RESOLVE(AllocateCommandBuffers);
        MI_VK_RESOLVE(FreeCommandBuffers);
        MI_VK_RESOLVE(BeginCommandBuffer);
        MI_VK_RESOLVE(EndCommandBuffer);
        MI_VK_RESOLVE(ResetCommandBuffer);
        MI_VK_RESOLVE(QueueSubmit);
        MI_VK_RESOLVE(CreateFence);
        MI_VK_RESOLVE(DestroyFence);
        MI_VK_RESOLVE(WaitForFences);
        MI_VK_RESOLVE(ResetFences);
        MI_VK_RESOLVE(CmdPipelineBarrier);
        MI_VK_RESOLVE(CreateQueryPool);
        MI_VK_RESOLVE(DestroyQueryPool);
        MI_VK_RESOLVE(GetQueryPoolResults);
        MI_VK_RESOLVE(CmdResetQueryPool);
        MI_VK_RESOLVE(CreateAccelerationStructureKHR);
        MI_VK_RESOLVE(DestroyAccelerationStructureKHR);
        MI_VK_RESOLVE(GetAccelerationStructureBuildSizesKHR);
        MI_VK_RESOLVE(GetAccelerationStructureDeviceAddressKHR);
        MI_VK_RESOLVE(CmdBuildAccelerationStructuresKHR);
        MI_VK_RESOLVE(CmdWriteAccelerationStructuresPropertiesKHR);
        MI_VK_RESOLVE(CmdCopyAccelerationStructureKHR);
        #undef MI_VK_RESOLVE
    }
};

static void vk_check(int32_t rv, const char *op) {
    if (rv != 0)
        Throw("VulkanAccel: %s failed with VkResult %d.", op, rv);
}

static VkDeviceAddress align_up(VkDeviceAddress addr, VkDeviceAddress align) {
    return (addr + (align - 1)) & ~(align - 1);
}

static VkDeviceAddress lookup_device_address(const void *ptr, const char *what) {
    VkDeviceAddress addr = jit_vulkan_device_address((void *) ptr);
    if (!addr)
        Throw("VulkanAccel: could not find the Vulkan buffer backing the "
              "%s data. The corresponding Dr.Jit array must be evaluated "
              "before building the acceleration structure.", what);
    return addr;
}

} // namespace

struct VulkanAccelData {
    VkDevice device = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroy_as = nullptr;
    VkAccelerationStructureKHR tlas = nullptr;
    void *tlas_buffer = nullptr;
    std::vector<VkAccelerationStructureKHR> blas_handles;
    std::vector<void *> blas_buffers;
    std::vector<JitIsectBinding *> bindings;

    ~VulkanAccelData() {
        for (JitIsectBinding *binding : bindings)
            jit_isect_unbind(binding);
        if (tlas && destroy_as)
            destroy_as(device, tlas, nullptr);
        if (tlas_buffer)
            jit_free(tlas_buffer);
        for (VkAccelerationStructureKHR b : blas_handles) {
            if (b && destroy_as)
                destroy_as(device, b, nullptr);
        }
        for (void *buf : blas_buffers) {
            if (buf)
                jit_free(buf);
        }
    }
};

std::pair<VulkanAccelData *, uint32_t>
build_vulkan_accel(const SceneIR &sd, const std::vector<uint32_t> &user_ids,
                   bool compact) {
    const std::vector<BlasEntry> &blases = sd.blases;
    const std::vector<InstanceEntry> &instances = sd.instances;

    if (sd.has_motion)
        Throw("VulkanAccel: animated instance transforms (motion blur) "
              "are not supported by the Vulkan acceleration structure backend.");

    for (const BlasEntry &blas : blases) {
        for (const ShapeIR &g : blas.geoms) {
            if (g.kind == ShapeIR::Kind::BSplineCurve ||
                g.kind == ShapeIR::Kind::LinearCurve)
                Throw("VulkanAccel: curve geometries (BSplineCurve / LinearCurve) "
                      "are not supported by the Vulkan acceleration structure backend.");
        }
    }

    // Ensure all prior Dr.Jit kernels/transfers filling geometry buffers have completed
    jit_sync_thread();

    VkDevice device = jit_vulkan_device_handle();
    VkQueue queue   = jit_vulkan_queue();
    int queue_fam   = jit_vulkan_queue_family();
    if (!device || !queue || queue_fam < 0)
        Throw("VulkanAccel: Vulkan backend is not initialized.");

    VulkanFunctions fn;
    fn.resolve();
    if (!fn.CreateAccelerationStructureKHR ||
        !fn.DestroyAccelerationStructureKHR ||
        !fn.GetAccelerationStructureBuildSizesKHR ||
        !fn.GetAccelerationStructureDeviceAddressKHR ||
        !fn.CmdBuildAccelerationStructuresKHR) {
        Throw("VulkanAccel: the active Vulkan device does not support "
              "VK_KHR_acceleration_structure.");
    }

    size_t n_blas = blases.size();
    bool do_compact = compact && n_blas > 0 &&
                      fn.CmdWriteAccelerationStructuresPropertiesKHR &&
                      fn.CmdCopyAccelerationStructureKHR &&
                      fn.CreateQueryPool && fn.DestroyQueryPool &&
                      fn.GetQueryPoolResults && fn.CmdResetQueryPool;

    auto accel = std::make_unique<VulkanAccelData>();
    accel->device = device;
    accel->destroy_as = fn.DestroyAccelerationStructureKHR;

    VkCommandPool cmd_pool = nullptr;
    VkCommandBuffer cmd = nullptr;
    VkFence fence = nullptr;
    VkQueryPool query_pool = nullptr;

    std::vector<void *> temp_buffers;
    std::vector<VkAccelerationStructureKHR> uncompacted_handles;
    std::vector<void *> uncompacted_buffers;

    auto cleanup_temp = [&]() {
        if (query_pool) {
            fn.DestroyQueryPool(device, query_pool, nullptr);
            query_pool = nullptr;
        }
        if (fence) {
            fn.DestroyFence(device, fence, nullptr);
            fence = nullptr;
        }
        if (cmd && cmd_pool) {
            fn.FreeCommandBuffers(device, cmd_pool, 1, &cmd);
            cmd = nullptr;
        }
        if (cmd_pool) {
            fn.DestroyCommandPool(device, cmd_pool, nullptr);
            cmd_pool = nullptr;
        }
        for (VkAccelerationStructureKHR h : uncompacted_handles) {
            if (h)
                fn.DestroyAccelerationStructureKHR(device, h, nullptr);
        }
        uncompacted_handles.clear();
        for (void *p : uncompacted_buffers) {
            if (p)
                jit_free(p);
        }
        uncompacted_buffers.clear();
        for (void *p : temp_buffers) {
            if (p)
                jit_free(p);
        }
        temp_buffers.clear();
    };

    try {
        VkCommandPoolCreateInfo pool_info {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                          VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = (uint32_t) queue_fam;
        vk_check(fn.CreateCommandPool(device, &pool_info, nullptr, &cmd_pool),
                 "vkCreateCommandPool");

        VkCommandBufferAllocateInfo alloc_info {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = cmd_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        vk_check(fn.AllocateCommandBuffers(device, &alloc_info, &cmd),
                 "vkAllocateCommandBuffers");

        VkFenceCreateInfo fence_info {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vk_check(fn.CreateFence(device, &fence_info, nullptr, &fence),
                 "vkCreateFence");

        auto submit_and_wait = [&]() {
            vk_check(fn.EndCommandBuffer(cmd), "vkEndCommandBuffer");
            VkSubmitInfo submit_info {};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &cmd;
            vk_check(fn.QueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
            vk_check(fn.WaitForFences(device, 1, &fence, VK_TRUE_VAL, UINT64_MAX),
                     "vkWaitForFences");
            vk_check(fn.ResetFences(device, 1, &fence), "vkResetFences");
            vk_check(fn.ResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        };

        // --------------------------------------------------------------------
        // 1. Pre-pass over BLASes and prepare geometry descriptors
        // --------------------------------------------------------------------
        bool any_backface_culled_triangles = false;
        std::vector<bool> blas_backface_cull(n_blas, false);
        std::vector<uint32_t> blas_ift_base(n_blas, 0u);
        uint32_t n_isect = 0;
        size_t aabb_total = 0;

        for (size_t blas_idx = 0; blas_idx < n_blas; ++blas_idx) {
            const BlasEntry &blas = blases[blas_idx];
            blas_ift_base[blas_idx] = n_isect;
            for (const ShapeIR &g : blas.geoms) {
                if (g.kind == ShapeIR::Kind::TrianglesCulled) {
                    any_backface_culled_triangles = true;
                    blas_backface_cull[blas_idx] = true;
                }
                if (g.kind == ShapeIR::Kind::Custom) {
                    n_isect++;
                    if (!g.aabb_buffer)
                        aabb_total += g.prim_count;
                }
            }
        }

        void *aabb_pool_ptr = nullptr;
        size_t aabb_cursor = 0;
        if (aabb_total > 0) {
            aabb_pool_ptr = jit_malloc(
                JitBackend::Vulkan, aabb_total * 6 * sizeof(float), /*shared=*/true);
            temp_buffers.push_back(aabb_pool_ptr);
        }

        std::vector<std::vector<VkAccelerationStructureGeometryKHR>> blas_geoms(n_blas);
        std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>> blas_ranges(n_blas);
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR *> blas_range_ptrs(n_blas, nullptr);
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> blas_build_infos(n_blas);
        std::vector<VkDeviceSize> blas_as_sizes(n_blas, 0);
        std::vector<VkDeviceSize> blas_scratch_sizes(n_blas, 0);

        for (size_t blas_idx = 0; blas_idx < n_blas; ++blas_idx) {
            const BlasEntry &blas = blases[blas_idx];
            if (blas.geoms.empty())
                Throw("VulkanAccel: BLAS %zu has no geometries.", blas_idx);

            auto &geoms  = blas_geoms[blas_idx];
            auto &ranges = blas_ranges[blas_idx];
            std::vector<uint32_t> max_prim_counts;

            geoms.reserve(blas.geoms.size());
            ranges.reserve(blas.geoms.size());
            max_prim_counts.reserve(blas.geoms.size());

            for (const ShapeIR &g : blas.geoms) {
                switch (g.kind) {
                    case ShapeIR::Kind::Triangles:
                    case ShapeIR::Kind::TrianglesCulled: {
                        Assert(g.index_stride == 3 * sizeof(uint32_t));
                        VkDeviceAddress v_addr =
                            lookup_device_address(g.vertex_ptr, "mesh vertex");
                        VkDeviceAddress i_addr =
                            lookup_device_address(g.index_ptr, "mesh index");

                        VkAccelerationStructureGeometryKHR geom {};
                        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

                        auto &tri = geom.geometry.triangles;
                        tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                        tri.vertexData.deviceAddress = v_addr;
                        tri.vertexStride = g.vertex_stride;
                        tri.maxVertex = g.vertex_count > 0 ? (uint32_t) (g.vertex_count - 1) : 0;
                        tri.indexType = VK_INDEX_TYPE_UINT32;
                        tri.indexData.deviceAddress = i_addr;

                        VkAccelerationStructureBuildRangeInfoKHR range {};
                        range.primitiveCount = (uint32_t) g.face_count;

                        geoms.push_back(geom);
                        ranges.push_back(range);
                        max_prim_counts.push_back((uint32_t) g.face_count);
                        break;
                    }

                    case ShapeIR::Kind::Custom: {
                        if (g.prim_count == 0)
                            Throw("VulkanAccel: bounding-box geometry with zero AABBs.");

                        VkDeviceAddress aabb_addr = 0;
                        if (g.aabb_buffer) {
                            aabb_addr = lookup_device_address(g.aabb_buffer, "bounding box");
                        } else {
                            void *dst = (uint8_t *) aabb_pool_ptr +
                                        aabb_cursor * 6 * sizeof(float);
                            g.fill_aabbs(g.ctx, dst);
                            aabb_addr = lookup_device_address(dst, "bounding box");
                            aabb_cursor += g.prim_count;
                        }

                        VkAccelerationStructureGeometryKHR geom {};
                        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                        geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
                        geom.flags = 0; // non-opaque so candidate hits invoke custom callback

                        auto &aabbs = geom.geometry.aabbs;
                        aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
                        aabbs.data.deviceAddress = aabb_addr;
                        aabbs.stride = 6 * sizeof(float);

                        VkAccelerationStructureBuildRangeInfoKHR range {};
                        range.primitiveCount = (uint32_t) g.prim_count;

                        geoms.push_back(geom);
                        ranges.push_back(range);
                        max_prim_counts.push_back((uint32_t) g.prim_count);
                        break;
                    }

                    case ShapeIR::Kind::BSplineCurve:
                    case ShapeIR::Kind::LinearCurve:
                        Throw("VulkanAccel: curve geometries are not supported.");

                    case ShapeIR::Kind::Instance:
                        Throw("VulkanAccel: instance geometry must be flattened "
                              "before reaching the BLAS builder.");
                }
            }

            VkFlags build_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            if (do_compact)
                build_flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;

            VkAccelerationStructureBuildGeometryInfoKHR &bi = blas_build_infos[blas_idx];
            memset(&bi, 0, sizeof(bi));
            bi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            bi.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            bi.flags = build_flags;
            bi.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            bi.geometryCount = (uint32_t) geoms.size();
            bi.pGeometries   = geoms.data();

            VkAccelerationStructureBuildSizesInfoKHR sizes {};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            fn.GetAccelerationStructureBuildSizesKHR(
                device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &bi, max_prim_counts.data(), &sizes);

            blas_as_sizes[blas_idx] =
                std::max<VkDeviceSize>(sizes.accelerationStructureSize, 256);
            blas_scratch_sizes[blas_idx] =
                std::max<VkDeviceSize>(sizes.buildScratchSize, 256);
        }

        // --------------------------------------------------------------------
        // 2. Build BLASes (and optionally compact them)
        // --------------------------------------------------------------------
        if (n_blas > 0) {
            auto &target_handles = do_compact ? uncompacted_handles : accel->blas_handles;
            auto &target_buffers = do_compact ? uncompacted_buffers : accel->blas_buffers;
            target_handles.resize(n_blas, nullptr);
            target_buffers.resize(n_blas, nullptr);

            for (size_t i = 0; i < n_blas; ++i) {
                void *as_mem = jit_malloc(JitBackend::Vulkan, blas_as_sizes[i], /*shared=*/false);
                target_buffers[i] = as_mem;
                size_t as_offset = 0;
                VkBuffer as_buf = (VkBuffer) jit_vulkan_lookup_buffer(as_mem, &as_offset);

                VkAccelerationStructureCreateInfoKHR create_info {};
                create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                create_info.buffer = as_buf;
                create_info.offset = as_offset;
                create_info.size   = blas_as_sizes[i];
                create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                vk_check(fn.CreateAccelerationStructureKHR(
                             device, &create_info, nullptr, &target_handles[i]),
                         "vkCreateAccelerationStructureKHR (BLAS)");

                void *scratch_mem = jit_malloc(
                    JitBackend::Vulkan, blas_scratch_sizes[i] + 256, /*shared=*/false);
                temp_buffers.push_back(scratch_mem);

                blas_build_infos[i].dstAccelerationStructure = target_handles[i];
                blas_build_infos[i].scratchData.deviceAddress =
                    align_up(jit_vulkan_device_address(scratch_mem), 256);
                blas_range_ptrs[i] = blas_ranges[i].data();
            }

            if (do_compact) {
                VkQueryPoolCreateInfo qp_info {};
                qp_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qp_info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
                qp_info.queryCount = (uint32_t) n_blas;
                vk_check(fn.CreateQueryPool(device, &qp_info, nullptr, &query_pool),
                         "vkCreateQueryPool");
            }

            VkCommandBufferBeginInfo begin_info {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vk_check(fn.BeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer");

            if (do_compact)
                fn.CmdResetQueryPool(cmd, query_pool, 0, (uint32_t) n_blas);

            fn.CmdBuildAccelerationStructuresKHR(
                cmd, (uint32_t) n_blas, blas_build_infos.data(), blas_range_ptrs.data());

            if (do_compact) {
                VkMemoryBarrier barrier {};
                barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                fn.CmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    0, 1, &barrier, 0, nullptr, 0, nullptr);

                fn.CmdWriteAccelerationStructuresPropertiesKHR(
                    cmd, (uint32_t) n_blas, uncompacted_handles.data(),
                    VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                    query_pool, 0);
            }

            submit_and_wait();

            if (do_compact) {
                std::vector<VkDeviceSize> compacted_sizes(n_blas, 0);
                vk_check(fn.GetQueryPoolResults(
                             device, query_pool, 0, (uint32_t) n_blas,
                             n_blas * sizeof(VkDeviceSize), compacted_sizes.data(),
                             sizeof(VkDeviceSize),
                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                         "vkGetQueryPoolResults");

                accel->blas_handles.resize(n_blas, nullptr);
                accel->blas_buffers.resize(n_blas, nullptr);

                bool any_shrunk = false;
                VkCommandBufferBeginInfo compact_begin {};
                compact_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                compact_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vk_check(fn.BeginCommandBuffer(cmd, &compact_begin),
                         "vkBeginCommandBuffer (compact)");

                for (size_t i = 0; i < n_blas; ++i) {
                    if (compacted_sizes[i] == 0 || compacted_sizes[i] >= blas_as_sizes[i]) {
                        accel->blas_handles[i] = uncompacted_handles[i];
                        accel->blas_buffers[i] = uncompacted_buffers[i];
                        uncompacted_handles[i] = nullptr;
                        uncompacted_buffers[i] = nullptr;
                        continue;
                    }

                    any_shrunk = true;
                    VkDeviceSize csize = std::max<VkDeviceSize>(compacted_sizes[i], 256);
                    void *c_mem = jit_malloc(JitBackend::Vulkan, csize, /*shared=*/false);
                    accel->blas_buffers[i] = c_mem;
                    size_t c_offset = 0;
                    VkBuffer c_buf = (VkBuffer) jit_vulkan_lookup_buffer(c_mem, &c_offset);

                    VkAccelerationStructureCreateInfoKHR create_info {};
                    create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                    create_info.buffer = c_buf;
                    create_info.offset = c_offset;
                    create_info.size   = csize;
                    create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                    vk_check(fn.CreateAccelerationStructureKHR(
                                 device, &create_info, nullptr, &accel->blas_handles[i]),
                             "vkCreateAccelerationStructureKHR (compacted BLAS)");

                    VkCopyAccelerationStructureInfoKHR copy_info {};
                    copy_info.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
                    copy_info.src   = uncompacted_handles[i];
                    copy_info.dst   = accel->blas_handles[i];
                    copy_info.mode  = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
                    fn.CmdCopyAccelerationStructureKHR(cmd, &copy_info);
                }

                if (any_shrunk) {
                    submit_and_wait();
                } else {
                    vk_check(fn.EndCommandBuffer(cmd), "vkEndCommandBuffer");
                    vk_check(fn.ResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
                }
            }
        }

        if (instances.empty())
            Throw("VulkanAccel: scene description contains no instances.");

        // --------------------------------------------------------------------
        // 3. Query BLAS device addresses & build TLAS over all instances
        // --------------------------------------------------------------------
        std::vector<VkDeviceAddress> blas_addresses(n_blas, 0);
        for (size_t i = 0; i < n_blas; ++i) {
            VkAccelerationStructureDeviceAddressInfoKHR addr_info {};
            addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addr_info.accelerationStructure = accel->blas_handles[i];
            blas_addresses[i] = fn.GetAccelerationStructureDeviceAddressKHR(device, &addr_info);
            if (!blas_addresses[i])
                Throw("VulkanAccel: failed to query BLAS device address.");
        }

        size_t n_inst = instances.size();
        VkAccelerationStructureInstanceKHR *inst_buf =
            (VkAccelerationStructureInstanceKHR *) jit_malloc(
                JitBackend::Vulkan,
                n_inst * sizeof(VkAccelerationStructureInstanceKHR),
                /*shared=*/true);
        temp_buffers.push_back(inst_buf);

        for (size_t i = 0; i < n_inst; ++i) {
            const InstanceEntry &inst = instances[i];
            VkAccelerationStructureInstanceKHR &vk_inst = inst_buf[i];
            memset(&vk_inst, 0, sizeof(VkAccelerationStructureInstanceKHR));

            // InstanceEntry::to_world is column-major 3x4; VkTransformMatrixKHR is row-major 3x4
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    vk_inst.transform.matrix[r][c] = inst.to_world[c * 3 + r];

            vk_inst.instanceCustomIndex = user_ids[i] & 0xFFFFFFu;
            vk_inst.mask = blases[inst.blas_index].visibility_mask & 0xFFu;
            vk_inst.instanceShaderBindingTableRecordOffset =
                blas_ift_base[inst.blas_index] & 0xFFFFFFu;

            uint32_t flags = 0u;
            if (any_backface_culled_triangles && !blas_backface_cull[inst.blas_index])
                flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            vk_inst.flags = flags & 0xFFu;

            vk_inst.accelerationStructureReference = blas_addresses[inst.blas_index];
        }

        VkAccelerationStructureGeometryKHR tlas_geom {};
        tlas_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tlas_geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlas_geom.flags = 0;
        tlas_geom.geometry.instances.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        tlas_geom.geometry.instances.arrayOfPointers = VK_FALSE_VAL;
        tlas_geom.geometry.instances.data.deviceAddress =
            jit_vulkan_device_address(inst_buf);

        VkAccelerationStructureBuildGeometryInfoKHR tlas_build_info {};
        tlas_build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        tlas_build_info.type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlas_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tlas_build_info.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlas_build_info.geometryCount = 1;
        tlas_build_info.pGeometries   = &tlas_geom;

        uint32_t tlas_prim_count = (uint32_t) n_inst;
        VkAccelerationStructureBuildSizesInfoKHR tlas_sizes {};
        tlas_sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        fn.GetAccelerationStructureBuildSizesKHR(
            device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &tlas_build_info, &tlas_prim_count, &tlas_sizes);

        VkDeviceSize tlas_as_size =
            std::max<VkDeviceSize>(tlas_sizes.accelerationStructureSize, 256);
        VkDeviceSize tlas_scratch_size =
            std::max<VkDeviceSize>(tlas_sizes.buildScratchSize, 256);

        accel->tlas_buffer = jit_malloc(JitBackend::Vulkan, tlas_as_size, /*shared=*/false);
        size_t tlas_offset = 0;
        VkBuffer tlas_buf = (VkBuffer) jit_vulkan_lookup_buffer(accel->tlas_buffer, &tlas_offset);

        VkAccelerationStructureCreateInfoKHR tlas_create_info {};
        tlas_create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        tlas_create_info.buffer = tlas_buf;
        tlas_create_info.offset = tlas_offset;
        tlas_create_info.size   = tlas_as_size;
        tlas_create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        vk_check(fn.CreateAccelerationStructureKHR(
                     device, &tlas_create_info, nullptr, &accel->tlas),
                 "vkCreateAccelerationStructureKHR (TLAS)");

        void *tlas_scratch = jit_malloc(
            JitBackend::Vulkan, tlas_scratch_size + 256, /*shared=*/false);
        temp_buffers.push_back(tlas_scratch);

        tlas_build_info.dstAccelerationStructure = accel->tlas;
        tlas_build_info.scratchData.deviceAddress =
            align_up(jit_vulkan_device_address(tlas_scratch), 256);

        VkAccelerationStructureBuildRangeInfoKHR tlas_range {};
        tlas_range.primitiveCount = tlas_prim_count;
        const VkAccelerationStructureBuildRangeInfoKHR *tlas_range_ptr = &tlas_range;

        VkCommandBufferBeginInfo tlas_begin {};
        tlas_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        tlas_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(fn.BeginCommandBuffer(cmd, &tlas_begin), "vkBeginCommandBuffer (TLAS)");

        fn.CmdBuildAccelerationStructuresKHR(cmd, 1, &tlas_build_info, &tlas_range_ptr);

        VkMemoryBarrier tlas_barrier {};
        tlas_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        tlas_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        tlas_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                     VK_ACCESS_SHADER_READ_BIT;
        fn.CmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &tlas_barrier, 0, nullptr, 0, nullptr);

        submit_and_wait();
        cleanup_temp();

        // --------------------------------------------------------------------
        // 4. Register the scene with Dr.Jit and bind custom intersection funcs
        // --------------------------------------------------------------------
        uint32_t geom_mask = 0x1u;
        if (n_isect) geom_mask |= 0x2u;
        if (any_backface_culled_triangles) geom_mask |= 0x8u;

        uint32_t scene_index = jit_vulkan_configure_scene(
            accel->tlas, n_isect, geom_mask);

        VulkanAccelData *accel_p = accel.release();
        jit_vulkan_scene_set_cleanup(
            scene_index,
            [](void *p) {
                auto *data = static_cast<VulkanAccelData *>(p);
                for (JitIsectBinding *b : data->bindings)
                    jit_isect_unbind(b);
                data->bindings.clear();
                jit_enqueue_host_func(
                    JitBackend::Vulkan,
                    [](void *q) { delete static_cast<VulkanAccelData *>(q); },
                    data);
            },
            accel_p);

        uint32_t ift_index = 0;
        for (const BlasEntry &blas : blases)
            for (const ShapeIR &g : blas.geoms)
                if (g.kind == ShapeIR::Kind::Custom)
                    accel_p->bindings.push_back(jit_isect_bind(
                        g.isect_func, scene_index, ift_index++, nullptr));

        Log(Debug, "VulkanAccel: built acceleration structures (%zu BLAS, "
                   "%zu instances, %u custom geometries)",
            accel_p->blas_handles.size(), instances.size(), n_isect);

        return { accel_p, scene_index };
    } catch (...) {
        cleanup_temp();
        throw;
    }
}

void release_vulkan_accel(VulkanAccelData *accel, uint32_t scene_index) {
    if (scene_index) {
        jit_sync_thread();
        jit_var_dec_ref(scene_index);
    } else {
        delete accel;
    }
}

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_VULKAN
