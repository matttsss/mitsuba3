# Dr.Jit Vulkan Backend (`JitBackend::Vulkan`) Implementation Roadmap

This roadmap tracks the end-to-end implementation of the **Vulkan backend** in `drjit-core` and `drjit`, using **textual SPIR-V code generation** assembled at runtime via `SPIRV-Tools` (`spvTextToBinary`) and **inline ray tracing queries** via `VK_KHR_acceleration_structure` and `VK_KHR_ray_query` (`SPV_KHR_ray_query`).

---

## Core Architectural Decisions

- **Textual SPIR-V + Runtime `SPIRV-Tools`**: Format textual SPIR-V assembly into `StringBuffer` (`src/strbuf.h`) and assemble it to `uint32_t` binary words at runtime using `spvTextToBinary` from `libSPIRV-Tools-shared`.
- **Dynamic Symbol Loading (`dlopen` / `LoadLibrary`)**: Dynamically load `libvulkan` and `libSPIRV-Tools-shared` at runtime in `src/vulkan_api.cpp` with self-contained header definitions in `src/vulkan_api.h` (`DRJIT_DYNAMIC_VULKAN`), requiring no build-time Vulkan SDK.
- **64-Bit Physical Storage Buffer Addressing (`PhysicalStorageBuffer64`)**: Use `OpMemoryModel PhysicalStorageBuffer64 GLSL450` (`SPV_KHR_physical_storage_buffer` + Vulkan 1.2 `bufferDeviceAddress`). Represent device allocations by their 64-bit `VkDeviceAddress` in `jitc_malloc`, backed by a side table mapping base `VkDeviceAddress -> {VkBuffer, VkDeviceMemory, size, mapped_ptr}`.
- **Inline Ray Tracing Queries (`VK_KHR_ray_query`)**: Mirror the Metal ray tracing API (`include/drjit-core/vulkan.h`: `jit_vulkan_configure_scene` + `jit_vulkan_ray_trace`), supporting both hardware triangle intersection and custom procedural AABB intersection functions (`jit_isect_*`) executed inside the `OpRayQueryProceedKHR` candidate loop.

---

## Milestone 1: Core Infrastructure, Dynamic Loader, Runtime & Memory Management

**Goal**: Initialize `JitBackend::Vulkan`, require Vulkan 1.3+ loader and devices, manage compute queues and timeline semaphores, and pass all memory allocation (`jitc_malloc`), host-device transfer (`jitc_memcpy` / `jitc_memcpy_async`), and synchronization tests.

### 1.1 Bitfields, Enums & Build System Wiring
> Done in commit `e508aa1` (branch `vulkan-backend` of `ext/drjit-core`). Verified by the new white-box test `tests/backend_bits.cpp` (`test_backend_bits`, 21776 checks) in the default, `DRJIT_ENABLE_VULKAN=OFF`, and `DRJIT_DYNAMIC_VULKAN=OFF` configurations; all existing LLVM/CUDA/OptiX tests unchanged w.r.t. the baseline.

- [x] Add `Vulkan = 4` (and update `Count = 5`) to `JitBackend` in `ext/drjit-core/include/drjit-core/jit.h` (C++ enum, C enum mirror, `jit_backend_name_v`; Vulkan is opt-in, i.e. not part of the default `jit_init()` set).
- [x] Widen `Variable::backend` from 2 bits to 3 bits and narrow `Variable::type` from 5 bits to 4 bits (`VarType::Count == 16`) in `ext/drjit-core/src/internal.h` (`kind:7 | backend:3 | type:4 | write_ptr:1 | written:1 = 16 bits`), preserving `sizeof(Variable) == 64` and `VariableKey` (static assertions added for the kind/backend/type widths).
- [x] Update the `VariableKey` type extraction bitshift in `jitc_shutdown()` (`ext/drjit-core/src/init.cpp`) from `(kv.first.packed >> 9) & 0x1f` to `(kv.first.packed >> 10) & 0xf`.
- [x] Widen `AllocInfo` backend bitfield in `ext/drjit-core/src/malloc.h` from `[size:53][shared:1][backend:2][device:8]` to `[size:52][shared:1][backend:3][device:8]` (`(data >> 8) & 0x7`).
- [x] Add `jitc_is_vulkan(JitBackend)` (and include Vulkan in `jitc_is_gpu()`), the `ts_vulkan` thread-local slot, and the `Kernel::vulkan` union member (the `Kernel` struct lives in `ext/drjit-core/src/io.h`). *Deviation*: no `ResourceKind::VulkanScene` is needed — `ResourceKind` is backend-agnostic, so Vulkan scenes will reuse `ResourceKind::Accel`.
- [x] Add `DRJIT_ENABLE_VULKAN` (default ON except on macOS) and `DRJIT_DYNAMIC_VULKAN` (default ON) options in `ext/drjit-core/CMakeLists.txt` with a `DRJIT_VULKAN_FILES` source list that grows with each sub-milestone (`vulkan_api.*` in 1.2, `vulkan_core.cpp`/`vulkan_ts.*` in 1.3, `vulkan_eval.*` in M2). `DRJIT_DYNAMIC_VULKAN=OFF` uses the official headers and links `Vulkan::Vulkan` (*changed in 1.2*: OFF only switches to the official headers; the loader is always loaded dynamically).

### 1.2 Dynamic Vulkan & SPIRV-Tools Loader (`src/vulkan_api.h`, `src/vulkan_api.cpp`)
> Done in commit `8d80ff5`. Verified by `test_vulkan_api` (268 checks on 2x RTX A6000 + lavapipe, 194 checks on lavapipe alone, no messages from the Khronos validation layer; graceful skip/failure without SPIRV-Tools or without a Vulkan driver) and `test_vulkan_layout` (1890 compile-time + 6 runtime checks against the official Vulkan 1.4.341 and SPIRV-Tools headers; 18/18 deliberate header mutations caught) in the default and `DRJIT_DYNAMIC_VULKAN=OFF` configurations; `DRJIT_ENABLE_VULKAN=OFF` builds provide API stubs. `test_backend_bits` (1.1) and all existing LLVM/CUDA/OptiX tests are unchanged w.r.t. the baseline.

- [x] Create `ext/drjit-core/src/vulkan_api.h` with self-contained Vulkan core types, constants, and `VK_KHR_acceleration_structure` / `VK_KHR_ray_query` structs when `DRJIT_DYNAMIC_VULKAN` is defined (otherwise the official headers are included in `VK_NO_PROTOTYPES` mode). The SPIRV-Tools C API subset is always self-contained.
- [x] Implement `jitc_vulkan_api_init()` and `jitc_vulkan_api_shutdown()` in `ext/drjit-core/src/vulkan_api.cpp`, called from `jitc_init()` / `jitc_shutdown()` when the Vulkan backend is requested. *Deviation*: `jitc_vulkan_api_init()` also creates the `VkInstance` (API version 1.3; `VK_KHR_portability_enumeration` when available) and the SPIRV-Tools context, since instance-level entry points must be resolved on an instance. Any failure logs a warning, releases all partial state, and only disables the Vulkan backend.
- [x] Dynamically load `libvulkan.so.1` (Linux), `vulkan-1.dll` (Windows), or `libvulkan.1.dylib` / `libMoltenVK.dylib` (macOS) and resolve `vkGetInstanceProcAddr` (search order: `DRJIT_LIBVULKAN_PATH`, default search path + glob, `$VULKAN_SDK`, alternative names); requires a Vulkan >= 1.3 loader.
- [x] Resolve instance-level Vulkan functions (`vkCreateInstance`, `vkDestroyInstance`, `vkEnumeratePhysicalDevices`, `vkGetPhysicalDeviceProperties2`, `vkGetPhysicalDeviceFeatures2`, `vkGetPhysicalDeviceQueueFamilyProperties`, `vkGetPhysicalDeviceMemoryProperties`, `vkCreateDevice`, `vkGetDeviceProcAddr`), plus the global functions `vkEnumerateInstanceVersion`/`vkEnumerateInstance{Extension,Layer}Properties` and `vkGetPhysicalDeviceProperties`, `vkEnumerateDeviceExtensionProperties`.
- [x] Resolve device-level Vulkan functions (`vkDestroyDevice`, `vkGetDeviceQueue`, `vkCreateCommandPool`, `vkDestroyCommandPool`, `vkResetCommandPool`, `vkAllocateCommandBuffers`, `vkFreeCommandBuffers`, `vkBeginCommandBuffer`, `vkEndCommandBuffer`, `vkQueueSubmit`, `vkQueueWaitIdle`, `vkCreateSemaphore`, `vkDestroySemaphore`, `vkWaitSemaphores`, `vkGetSemaphoreCounterValue`, `vkCreateBuffer`, `vkDestroyBuffer`, `vkGetBufferMemoryRequirements`, `vkGetBufferDeviceAddress`, `vkAllocateMemory`, `vkFreeMemory`, `vkBindBufferMemory`, `vkMapMemory`, `vkUnmapMemory`, `vkCreateShaderModule`, `vkDestroyShaderModule`, `vkCreatePipelineLayout`, `vkDestroyPipelineLayout`, `vkCreateComputePipelines`, `vkDestroyPipeline`, `vkCreatePipelineCache`, `vkDestroyPipelineCache`, `vkGetPipelineCacheData`, `vkCmdBindPipeline`, `vkCmdPushConstants`, `vkCmdDispatch`, `vkCmdPipelineBarrier`, `vkCmdCopyBuffer`, `vkCmdFillBuffer`), plus `vkDeviceWaitIdle`, `vkResetCommandBuffer`, `vkSignalSemaphore`, `vkFlushMappedMemoryRanges`, `vkInvalidateMappedMemoryRanges`, `vkCmdUpdateBuffer`, and the optional `VK_KHR_acceleration_structure` entry points (`jitc_vulkan_has_accel_api`).
- [x] Dynamically load `libSPIRV-Tools-shared.{so,dll,dylib}` and resolve `spvContextCreate`, `spvContextDestroy`, `spvTextToBinary`, `spvBinaryDestroy`, and `spvDiagnosticDestroy` (search order: `DRJIT_LIBSPIRV_TOOLS_PATH`, default search path + glob, `$VULKAN_SDK`, `libSPIRV-Tools.so`), plus the optional `spvBinaryToText`, `spvTextDestroy`, `spvValidateBinary`, and `spvSoftwareVersionString`.
- [x] *Added*: public API `jit_vulkan_version()`, `jit_vulkan_instance()`, and `jit_vulkan_lookup()` in `include/drjit-core/jit.h` (stubs when `DRJIT_ENABLE_VULKAN=OFF`). `jit_has_backend(JitBackend::Vulkan)` stays false until devices are supported (1.3).
- [x] *Added*: tests `tests/vulkan_api.cpp` (`test_vulkan_api`) and `tests/vulkan_layout.cpp` (`test_vulkan_layout`, generated by `tests/vulkan_layout_gen.py`, only built when the official Vulkan headers are found; SPIRV-Tools checks when `spirv-tools/libspirv.h` is found as well).

### 1.3 Vulkan Device Context, Command Submission & Memory Allocator (`src/vulkan.h`, `src/vulkan_core.cpp`, `src/vulkan_ts.cpp`)
> Done in commit `8649adb`. Verified by `tests/vulkan.cpp` (17/17 tests passing on NVIDIA RTX A6000 and lavapipe; clean runs under `VK_LAYER_KHRONOS_validation` + `VK_LAYER_VALIDATE_SYNC=1` with 0 errors/warnings/hazards); `test_vulkan_api` (287/287 passing); `test_vulkan_layout` (1896 passing); `test_backend_bits` (21776 passing); `novk_stubs` API stubs test passing against `DRJIT_ENABLE_VULKAN=OFF`; clean builds with `DRJIT_DYNAMIC_VULKAN=OFF`; zero regressions across existing test suites (`basics`, `mem`, `loop`, `vcall`, `reductions`, `array`, `record`, `half`).

- [x] Implement `jitc_vulkan_init()` and `jitc_vulkan_shutdown()` in `ext/drjit-core/src/vulkan_core.cpp` and wire them into `jitc_init()` / `jitc_shutdown()` in `ext/drjit-core/src/init.cpp`.
- [x] Enumerate physical devices supporting compute queues, `bufferDeviceAddress`, `timelineSemaphore`, and `scalarBlockLayout` (the `VkInstance` itself is already created in 1.2). Automatic device selection prioritizes discrete > integrated > virtual > other > CPU; overrideable via `DRJIT_VULKAN_DEVICE=<index>`.
- [x] Query and store per-device capability flags (`shaderInt8`, `shaderInt16`, `shaderInt64`, `shaderFloat16`, `shaderFloat64`, `subgroupSize`, `rayQuery`, `accelerationStructure`, `shaderBufferFloat32AtomicAdd`). *Deviation*: `shaderInt64` is required for 64-bit `PhysicalStorageBuffer64` pointer arithmetic, rather than optional.
- [x] Implement `VulkanThreadState` (subclassing `ThreadState`) in `ext/drjit-core/src/vulkan_ts.cpp`:
  - [x] Single shared command stream (`VulkanStream`) per device recorded under `state.lock` (replacing per-thread command buffers to eliminate cross-thread unsubmitted-work visibility hazards).
  - [x] Automatic read/write hazard tracking (RAW/WAR/WAW on up to 16 ranges) and global memory barrier insertion.
  - [x] Timeline semaphore (`VkSemaphore`) submission and `sync()` (`vkWaitSemaphores`).
  - [x] Host callbacks (`jit_enqueue_host_func`): enqueued on timeline semaphore and dispatched during synchronization on the calling thread (or immediately if idle). Callbacks execute outside `state.lock`. *Follow-up noted*: a dedicated background waiter thread blocking on `vkWaitSemaphores` will be introduced in a future milestone for asynchronous callback execution.
- [x] Integrate Vulkan memory allocation with `jitc_malloc` in `ext/drjit-core/src/malloc.cpp`:
  - [x] Device-local allocations: Allocate `VkDeviceMemory` (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` + `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`), bind dedicated `VkBuffer`, query `VkDeviceAddress`, record in buffer registry, and return `(void *)(uintptr_t) device_address` as Dr.Jit pointer. Dual-layer collision safety via `state.alloc_used` (exact base) and buffer registry (interior overlap). Oversize pre-check against `maxMemoryAllocationSize` (due to power-of-two rounding, effective cap is 2 GiB on NVIDIA RTX A6000).
  - [x] Shared memory allocations: Allocate host-visible coherent memory (`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`, preferring `HOST_CACHED_BIT`), map persistent CPU pointer, and register. Shared frees deferred via `jitc_vulkan_free_later` until the GPU signals completion.
  - [x] Implement `jitc_memcpy`, `jitc_memcpy_async`, and `jitc_memset_async` using staging buffers and `vkCmdCopyBuffer` / `vkCmdFillBuffer` / `vkCmdUpdateBuffer`. Also implemented `jit_poke()` without kernels in 1.3 (pulled forward from 3.3) for scalar writes.
- [x] *Added*: Public device query API in `include/drjit-core/jit.h` and `src/api.cpp` (`jit_vulkan_device_handle`, `jit_vulkan_physical_device`, `jit_vulkan_queue`, `jit_vulkan_queue_family`, `jit_vulkan_device_index`, `jit_vulkan_device_name`, `jit_vulkan_lookup_buffer`, `jit_vulkan_device_address`).
- [x] *Added*: Test harness additions in `tests/test.h` and `tests/test.cpp` (`VulkanArray` aliases, `TEST_VULKAN`, `-k` flag to run only Vulkan tests, `DRJIT_TEST_REQUIRE_VULKAN=1` failure check, and fixed unhandled exception exit code).

### 1.4 Vulkan 1.3 Baseline & Core Maintenance4 Integration (`src/vulkan_api.h`, `src/vulkan_core.cpp`, `src/malloc.cpp`)
> Done in commit `09a48ca`. Verified by `test_vulkan_layout` (1,987 static and 6 runtime checks passed against official headers), `test_vulkan_api` (291/291 passed), `test_vulkan` (17/17 passed on NVIDIA RTX A6000 and Mesa llvmpipe; 100% clean under `VK_LAYER_VALIDATE_SYNC=1`); passing static and disabled builds (`novk_stubs: OK`); 0 regressions across all suites (`basics`, `mem`, `loop`, `vcall`, `reductions`, `array`, `record`, `half`, `backend_bits`).

- [x] Elevate minimum required Vulkan version from 1.2 to 1.3 across loader checks, instance creation, and physical device queries.
- [x] Add `VkPhysicalDeviceVulkan13Properties` (including `maxBufferSize`) to `src/vulkan_api.h` and update layout checks in `tests/vulkan_layout.cpp` (1,987 static checks).
- [x] Query and enable `VkPhysicalDeviceVulkan13Features` directly in the core device feature chain, enforcing `features13.maintenance4 = VK_TRUE` and enabling `features13.synchronization2`.
- [x] Target `SPV_ENV_VULKAN_1_3` in runtime SPIRV-Tools context creation (`spvContextCreate`).
- [x] Store `max_buffer_size` in `VulkanDevice` (`src/internal.h`) and validate allocation limits against $\min(\text{maxMemoryAllocationSize}, \text{maxBufferSize})$ in `src/malloc.cpp`.
- [x] Retain the power-of-two schema for memory allocations (`size = round_pow2(size)`), keeping allocation sizing uniform across all backends.
- [x] Update `tests/vulkan.cpp` to verify device API version $\ge 1.3$, 1 GiB allocation success, and that 3 GiB requests round to 4 GiB and cleanly raise the device allocation size limit exception.

### 1.5 Sparse Buffer Binding Fallback for Large Allocations (`src/vulkan_api.h`, `src/vulkan_core.cpp`, `src/malloc.cpp`)
> Done in commit `7cc7ad1`. Verified by `test_vulkan_layout` (2,027 static and 6 runtime checks passed), `test_vulkan_api` (291/291 passed), `test_vulkan` (17/17 passed on NVIDIA RTX A6000 and Mesa llvmpipe; 100% clean under `VK_LAYER_VALIDATE_SYNC=1`); 0 regressions across all suites (`basics`, `mem`, `loop`, `vcall`, `reductions`, `array`, `record`, `half`, `backend_bits`).

- [x] Relax memory allocation limit in `src/malloc.cpp`: allow device-local allocations up to `dev.max_buffer_size` (1 TiB via `maintenance4`) when sparse buffer binding is supported.
- [x] Add self-contained declarations for sparse memory binding to `src/vulkan_api.h` (`VkSparseMemoryBind`, `VkSparseBufferMemoryBindInfo`, `VkBindSparseInfo`, `vkQueueBindSparse`, etc.) and regenerate `tests/vulkan_layout.cpp` (2,027 static assertions).
- [x] In `vulkan_create_device`: enable `features.sparseBinding` and `features.sparseResidencyBuffer` if supported by the device and compute queue family.
- [x] In `vulkan_buffer_new`: attempt direct single allocation first. If `vkAllocateMemory` fails (e.g. `VK_ERROR_OUT_OF_DEVICE_MEMORY`) on allocations exceeding `maxMemoryAllocationSize` (~3.998 GiB on NVIDIA), automatically fall back to creating a sparse residency buffer (`VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT`) and binding multiple chunk allocations ($\le \text{max\_memory\_allocation\_size}$) via `vkQueueBindSparse`.
- [x] In `vulkan_buffer_free`: cleanly free all chunk handles stored in `sparse_memories`.
- [x] Update `tests/vulkan.cpp` to verify 4 GiB allocation (`(size_t) 3 << 30` rounded up) when supported by `maxBufferSize`, verify shared 4 GiB raises exception, and verify oversize bounds ($1 \ll 50$) still raise exception.

### 1.6 Shared-Backend Simplifications & Power-of-Two Invariant Checks (`src/vulkan_core.cpp`, `src/vulkan.h`, `src/vulkan_ts.cpp`, `src/malloc.cpp`)
> Done in commit `ccf59f5`. Verified by `test_vulkan_layout` (2,103 static and 6 runtime checks passed), `test_vulkan_api` (289/289 passed), `test_vulkan` (17/17 passed on NVIDIA RTX A6000 and Mesa llvmpipe; 100% clean under `VK_LAYER_VALIDATE_SYNC=1` with 0 hazards); 0 regressions across all suites (21,776 checks passed).

- [x] Streamline hazard tracking in `src/vulkan_core.cpp`, `src/vulkan.h`, and `src/vulkan_ts.cpp`: replace granular byte-range tracking (`VulkanRange`, `vulkan_overlaps`, dynamic vector reallocations) with coarse pipeline barriers (`jitc_vulkan_access(dev, bool is_write)`), aligning with CUDA/Metal stream execution semantics.
- [x] Add debug mode assertions (`jitc_assert((size & (size - 1)) == 0)`) in `jitc_malloc` and `vulkan_buffer_new` to strictly enforce the power-of-two allocation invariant.
- [x] Retain pointer collision handling (`jitc_malloc_collision` and `alloc_used` check) in `src/malloc.cpp`.
- [x] Simplify `max_size` check in `src/malloc.cpp` using guaranteed Vulkan 1.3 `dev.max_buffer_size`.
- [x] Update legacy comments in `src/vulkan_api.h` referring to Vulkan 1.2.

---

## Milestone 2: Basic SPIR-V Codegen & Compute Execution

**Goal**: Generate valid textual SPIR-V for branchless compute kernels, assemble via `spvTextToBinary`, create `VkPipeline` objects, and execute arithmetic, math, bitwise, cast, gather, and scatter operations.

### 2.1 Multi-Section SPIR-V Module Assembly (`src/vulkan_eval.h`, `src/vulkan_eval.cpp`, `src/strbuf.h`, `src/strbuf.cpp`, `src/var.h`, `src/var.cpp`, `src/eval.h`, `src/eval.cpp`)
> Done in commit `752a28d`. Verified by `test_vulkan` (`18_spirv_assemble` assembling & validating SPIR-V 1.6 modules via `spvTextToBinary` + `spvValidateBinary` under `SPV_ENV_VULKAN_1_3` across all 12 scalar types + `Pointer`, scalar vs. vector inputs, and both `PushConstant` and `PhysicalStorageBuffer` parameter modes; 18/18 passed on NVIDIA RTX A6000 and Mesa llvmpipe).

- [x] Extend `StringBuffer` in `ext/drjit-core/src/strbuf.h` and `src/strbuf.cpp` (`fmt_vulkan`) and add `type_name_vulkan` / `type_name_vulkan_bin` in `src/var.h` / `src/var.cpp` to format SPIR-V SSA register names (`%r<reg_index>`), SPIR-V type tokens (`%void`, `%bool`, `%i8`, `%u8`, `%i16`, `%u16`, `%i32`, `%u32`, `%i64`, `%u64`, `%f16`, `%f32`, `%f64`), literals (`$l`), alignments (`$a`), and parameter indices (`$o`).
- [x] Implement multi-section staging buffers in `ext/drjit-core/src/vulkan_eval.cpp` to satisfy SPIR-V 1.6's strict global ordering:
  - [x] Section 1–6 (`spv_header`): `OpCapability` (`Shader`, `Int64`, `PhysicalStorageBufferAddresses`, plus on-demand `Int8`, `StorageBuffer8BitAccess`, `Int16`, `StorageBuffer16BitAccess`, `Float16`, `Float64` verified against `VulkanDevice`), optional `%glsl450 = OpExtInstImport "GLSL.std.450"`, `OpMemoryModel PhysicalStorageBuffer64 GLSL450`, `OpEntryPoint GLCompute %main "drjit_^^^^..."`, `OpExecutionMode %main LocalSize <wg_size> 1 1`.
  - [x] Section 8 (`spv_annotations`): `OpDecorate` / `OpMemberDecorate` (`BuiltIn GlobalInvocationId`, `BuiltIn NumWorkgroups`, `Block`, `Offset`, `ArrayStride` for `PhysicalStorageBuffer` pointers).
  - [x] Section 9 (`spv_types_consts`): Deduplicated `OpType*`, `OpConstant*` (`OpConstantTrue`, `OpConstantFalse`, `OpConstantNull`, scalar literals), and module-scope `OpVariable` (`%gl_GlobalInvocationID`, `%gl_NumWorkGroups`, `%params` / `%pc`).
  - [x] Section 10 (`spv_func_locals` + `buffer`): Entry basic block `OpVariable ... Function` declarations followed by the scheduled SSA instruction stream.
- [x] Implement kernel parameter passing:
  - [x] Pass `size` (`Offset 0`) and `uint64_t args[]` (`Offset 8`, matching `kernel_params.data()` layout on 64-bit hosts) via `PushConstant` (`%params`) when `n_params * sizeof(void *) <= maxPushConstantsSize`.
  - [x] Fall back to a `PhysicalStorageBuffer` pointer (`%pc` holding `%ptr_psb_Params` at `Offset 0`, loaded into `%params` in `%entry`) when parameter footprint exceeds `maxPushConstantsSize`.
- [x] Emit the compute kernel grid-stride loop structure (`%entry` -> `%loop_head` with `OpPhi` and `OpLoopMerge` -> `%loop_body` -> `%loop_cont` -> `%loop_exit` -> `OpReturn`) and load/store `ParamType::Input` (scalar `size == 1` vs. vector `size > 1`, `Bool` 8-bit storage conversion, `Pointer` literals) and `ParamType::Output` variables.

### 2.2 Basic IR Operations (`jitc_vulkan_render` in `src/vulkan_eval.cpp`)
> Done. Verified by `test_vulkan` (`19_spirv_ops` assembling & validating SPIR-V 1.6 modules via `spvTextToBinary` + `spvValidateBinary` under `SPV_ENV_VULKAN_1_3` across all 12 scalar types, all arithmetic/math/comparison/bitwise/bit-manipulation operations, fused float-to-int rounding, `FastMath=0/1`, and all $12 \times 12 = 144$ pairwise `Cast`s; 19/19 passed on NVIDIA RTX A6000 and Mesa llvmpipe).

- [x] **Literals & Thread Index**:
  - [x] `VarKind::Literal`: Emit `OpConstantTrue` / `OpConstantFalse` / `OpConstant` (using `!0x...` raw bit-pattern words for float literals) in `spv_types_consts` and bind register.
  - [x] `VarKind::Counter`: Bind to the grid-stride loop induction variable `%idx` via `OpCopyObject`.
  - [x] `VarKind::Undefined`: Bind to `OpConstantNull %<type>` in `spv_types_consts`.
- [x] **Integer & Floating-Point Arithmetic**:
  - [x] `Neg`: `OpSNegate` / `OpFNegate`
  - [x] `Add`, `Sub`, `Mul`: `OpIAdd` / `OpFAdd`, `OpISub` / `OpFSub`, `OpIMul` / `OpFMul`
  - [x] `Div`, `DivApprox`, `Mod`: `OpSDiv` / `OpUDiv` / `OpFDiv`, `OpSRem` / `OpUMod`
  - [x] `MulHi`, `MulWide`: `OpSMulExtended` / `OpUMulExtended` + `OpCompositeExtract` (with `%struct_mul_<type> = OpTypeStruct %<type> %<type>`) and 64-bit widening multiply (`OpSConvert` / `OpUConvert` + `OpIMul`)
  - [x] `Fma`: `OpExtInst %t %glsl450 Fma` (float) / `OpIMul` + `OpIAdd` (integer)
  - [x] `Min`, `Max`, `FMin`, `FMax`, `Abs`, `Copysign`: `OpExtInst %t %glsl450 SMin/UMin/FMin`, `SMax/UMax/FMax` (with `OpIsNan` + `OpSelect` quiet-NaN propagation for `Min`/`Max` and `NMin`/`NMax` for `FMin`/`FMax`), `SAbs/FAbs` (`OpCopyObject` for unsigned), and bitwise sign-bit transplant for `Copysign`
  - [x] `Sqrt`, `SqrtApprox`, `Rcp`, `RcpApprox`, `RSqrtApprox`: `OpExtInst %t %glsl450 Sqrt`, `OpFDiv %t %c_one_<t> %x`, `OpExtInst %t %glsl450 InverseSqrt`
  - [x] `Ceil`, `Floor`, `Trunc`, `Round`: `OpExtInst %t %glsl450 Ceil`, `Floor`, `Trunc`, `RoundEven` (plus fused `OpConvertFToS` / `OpConvertFToU` when target type is an integer)
- [x] **Comparisons & Selection**:
  - [x] Integer comparisons (`Eq`, `Neq`, `Lt`, `Le`, `Gt`, `Ge`): `OpIEqual`, `OpINotEqual`, `OpSLessThan`/`OpULessThan`, `OpSLessThanEqual`/`OpULessThanEqual`, `OpSGreaterThan`/`OpUGreaterThan`, `OpSGreaterThanEqual`/`OpUGreaterThanEqual`
  - [x] Float comparisons: `OpFOrdEqual`, `OpFOrdNotEqual`, `OpFOrdLessThan`, `OpFOrdLessThanEqual`, `OpFOrdGreaterThan`, `OpFOrdGreaterThanEqual`
  - [x] Boolean comparisons: `OpLogicalEqual`, `OpLogicalNotEqual`, and `OpLogicalNot` + `OpLogicalAnd`/`OpLogicalOr` for ordered boolean comparisons
  - [x] `Select`: `OpSelect %t %cond %true_val %false_val`
- [x] **Bitwise, Logic & Bit Manipulation**:
  - [x] `Not`: `OpLogicalNot` (for `Bool`), `OpNot` (for integers), `OpBitcast` + `OpNot` (for floats)
  - [x] `And`, `Or`, `Xor`: `OpLogicalAnd`/`OpBitwiseAnd`, `OpLogicalOr`/`OpBitwiseOr`, `OpLogicalNotEqual`/`OpBitwiseXor` (including float bitcast and mixed `(reg, mask)` handling via `OpSelect` with `%c_zero_<t>` / `%c_ones_<t>`)
  - [x] `Shl`, `Shr`: `OpShiftLeftLogical`, `OpShiftRightArithmetic` (signed), `OpShiftRightLogical` (unsigned)
  - [x] `Popc`, `Brev`, `Clz`, `Ctz`: `OpBitCount`, `OpBitReverse`, `OpExtInst %glsl450 FindUMsb`, and `OpExtInst %glsl450 FindILsb` across 8/16/32/64-bit integers (promoting 8/16-bit and splitting 64-bit to 32-bit `%u32` as required by Vulkan `VUID-RuntimeSpirv-None-10824` and `GLSL.std.450`)
- [x] **Type Conversions & Transcendentals**:
  - [x] `Cast`: `OpSConvert`, `OpUConvert` (with intermediate unsigned/signed bitcast when widening across signedness boundaries), `OpFConvert`, `OpConvertFToS`, `OpConvertFToU`, `OpConvertSToF`, `OpConvertUToF`, and `Bool` <-> numeric conversions via `OpSelect` / `OpINotEqual` / `OpFOrdNotEqual`
  - [x] `Bitcast`: `OpBitcast` (or `OpCopyObject` when source and target SPIR-V types match)
  - [x] Transcendental math (`Sin`, `Cos`, `Exp2`, `Log2`, `Tanh`): `GLSL.std.450` extended instructions (`Sin`, `Cos`, `Exp2`, `Log2`, `Tanh`, upcasting/downcasting `Float64` through `%f32`), complementing Dr.Jit's default polynomial expansions in `src/op.cpp`.

### 2.3 Memory Operations (`Gather`, `Scatter`, `PacketGather`, `PacketScatter`, `ScatterReduce`)
- [x] `VarKind::Gather`: Emit masked conditional load (`OpSelectionMerge` + `OpBranchConditional` + `OpPtrAccessChain %ptr_psb_<T> %base %idx` + `OpLoad ... Aligned <sizeof(T)>` + `OpPhi`), converting `Bool` from `uint8_t` in memory.
- [x] `VarKind::PacketGather` & `VarKind::PacketScatter`: Emit wide vector (`v2`, `v4`) loads and stores using `OpTypeVector` physical storage buffer pointers with `Aligned <N*sizeof(T)>`.
- [x] `VarKind::Scatter`: Emit masked conditional store (`OpSelectionMerge` + `OpBranchConditional` + `OpPtrAccessChain` + `OpStore ... Aligned <sizeof(T)>`).
- [x] `VarKind::ScatterReduce` & `VarKind::ScatterInc`:
  - [x] Emit `OpAtomicIAdd`, `OpAtomicSMin`/`OpAtomicUMin`, `OpAtomicSMax`/`OpAtomicUMax`, `OpAtomicAnd`, `OpAtomicOr`, `OpAtomicXor` (plus `ScatterExch` via `OpAtomicExchange`, `ScatterCAS` via `OpAtomicCompareExchange`, and `BoundsCheck`).
  - [x] Support `Float32`/`Float64` atomic add via `OpAtomicFAddEXT` (`SPV_EXT_shader_atomic_float_add` when supported) with an integer `OpAtomicCompareExchange` CAS loop fallback, and float `Min`/`Max` via signed/unsigned integer atomics (`OpAtomicSMin`/`OpAtomicUMax`, `OpAtomicSMax`/`OpAtomicUMin`).
  - [x] Support subgroup pre-aggregation (`JitFlag::ScatterReduceLocal`) via `OpGroupNonUniformBallot` and `OpGroupNonUniformIAdd` / `OpGroupNonUniformFAdd` (and `Min`/`Max`/`BitwiseAnd`/`BitwiseOr`).

### 2.4 Runtime Compilation (`spvTextToBinary` -> `VkPipeline`) & Basic Verification
- [x] Implement `jitc_vulkan_kernel_compile()` in `ext/drjit-core/src/vulkan_eval.cpp` and `VulkanThreadState::launch()` in `ext/drjit-core/src/vulkan_ts.cpp`:
  - [x] Assemble textual SPIR-V via `spvTextToBinary`, reporting line-accurate diagnostics on failure.
  - [x] Create `VkShaderModule` and `VkPipeline` using a persistent `VkPipelineCache` serialized under `~/.drjit/`.
  - [x] Integrate with Dr.Jit's in-memory `unit_cache` and disk kernel cache (`CacheKind::VulkanSpirv`) in `ext/drjit-core/src/eval.cpp` and `ext/drjit-core/src/io.cpp`.
- [x] Add end-to-end kernel execution tests (`21_eval_arithmetic_and_casts`, `22_eval_memory_and_atomics`, `23_kernel_cache_and_history`) in `ext/drjit-core/tests/vulkan.cpp` testing arithmetic, math intrinsics, casts, gathers, scatters, atomics, subgroup reductions, and kernel caches.

---

## Milestone 3: Symbolic Control Flow, Polymorphic Calls & Utility Kernels

**Goal**: Support symbolic loops (`Loop`), symbolic conditionals (`If`), local arrays (`Array`), polymorphic virtual function calls (`Call`), and GPU utility kernels (`reduce`, `scan`, `mkperm`, `compress`).

### 3.1 Structured Control Flow (`If` and `While` Loops) & Local Arrays
- [x] **Symbolic Conditionals (`CondStart`, `CondMid`, `CondEnd`, `CondOutput`)**:
  - [x] Allocate function-local `%cond_out_<k> = OpVariable %ptr_func_<T> Function` in `spv_func_locals` for each active `CondOutput`.
  - [x] Emit `OpSelectionMerge %cond_merge_<id> None` + `OpBranchConditional %cond %cond_true_<id> %cond_false_<id>` at `CondStart`.
  - [x] Store true-branch values (`OpStore %cond_out_<k> %vt`) and branch to `%cond_merge_<id>` at `CondMid`.
  - [x] Store false-branch values (`OpStore %cond_out_<k> %vf`), branch to `%cond_merge_<id>`, and load `CondOutput` registers (`OpLoad`) at `CondEnd`.
- [x] **Symbolic Loops (`LoopStart`, `LoopPhi`, `LoopCond`, `LoopEnd`, `LoopOutput`)**:
  - [x] Allocate function-local `%loop_var_<k> = OpVariable %ptr_func_<T> Function` in `spv_func_locals` for each loop state variable and initialize with `outer_in` before `%loop_header_<id>`.
  - [x] Emit `%loop_header_<id> = OpLabel`, `OpLoopMerge %loop_exit_<id> %loop_cont_<id> None`, `OpBranch %loop_body_<id>`, and load `LoopPhi` registers at `LoopStart`.
  - [x] Emit `OpSelectionMerge %loop_cond_ok_<id> None` + `OpBranchConditional %cond %loop_cond_ok_<id> %loop_exit_<id>` at `LoopCond`.
  - [x] Store updated `inner_out` values to `%loop_var_<k>`, branch through `%loop_cont_<id>` back to `%loop_header_<id>`, emit `%loop_exit_<id> = OpLabel`, and load `LoopOutput` registers at `LoopEnd`.
- [x] **Local Arrays (`Array`, `ArrayInit`, `ArrayRead`, `ArrayWrite`, `ArrayPhi`, `ArraySelect`)**:
  - [x] Declare `OpTypeArray %elem_type %c_u32_<len>` and `%arr_<id> = OpVariable %ptr_func_arr Function` in `spv_func_locals`.
  - [x] Implement `ArrayInit` (loop or unrolled store), `ArrayRead` (`OpAccessChain` + `OpLoad`), `ArrayWrite` (masked `OpAccessChain` + `OpStore`), and `ArraySelect`.

### 3.2 Polymorphic Calls (`VarKind::Call`, `CallInput`, `CallOutput`, `CallSelf`, `CallGetter`)
> **Limitation (No Separate Per-Callable Driver Compilation)**: Unlike LLVM (separate `.o` object files + function pointers), CUDA (`cuLinkAddData` + `.global .u64 callables[]`), and Metal (`MTLLinkedFunctions` + `MTLVisibleFunctionTable`), standard Vulkan compute (`VK_SHADER_STAGE_COMPUTE_BIT`) takes a single `VkShaderModule` per pipeline, forbids `OpCapability Linkage` (`VUID-RuntimeSpirv-None-06275`), and lacks function-pointer tables in compute shaders. All callable `OpFunction`s (captured via `UnitBuilder`) must therefore be merged into the entry unit's SPIR-V module at the end of `jitc_vulkan_assemble()`, dispatched via `OpSwitch` over direct `OpFunctionCall`s, and compiled together in a single `vkCreateComputePipelines` call.

- [x] Implement `jitc_vulkan_assemble_func()` in `ext/drjit-core/src/vulkan_eval.cpp`:
  - [x] Emit each `CallableUnit` as a SPIR-V function `%func_<hash>` returning `%Tuple_<out_codes>` (or `%void`).
  - [x] Implement coalesced vector loads (`v4u32` / `v2u32` / `u32` via `PhysicalStorageBuffer`) from the callable's `%data_ptr` for captured variables and pointers.
- [x] Implement `jitc_var_call_assemble_vulkan()`:
  - [x] Load the 64-bit offset table entry from `call_buffer` when `has_slots` is true to compute `%call_dptr`.
  - [x] Emit direct `OpFunctionCall %ret_type %func_unique_<hash>` when `call->n_inst == 1`.
  - [x] Emit `OpSelectionMerge` + `OpSwitch %self %default [inst_id_0 %case_k0, ...]` with per-unique-target `OpFunctionCall %ret_type %func_<hash_k>` when `call->n_inst > 1`.
- [x] Implement `jitc_var_call_getter_assemble_vulkan()` for constant-offset getter loads from `call_buffer`.

### 3.3 Device Utility Kernels (`resources/vulkan_kernels.glsl`, `src/vulkan_ts.cpp`, `src/vulkan_core.cpp`)
- [x] Implement `VulkanThreadState::poke()` (8-bit, 16-bit, 32-bit, 64-bit scalar write kernel).
- [x] Implement `VulkanThreadState::aggregate()` (batched memcpy/fill kernel for `jit_aggregate`).
- [x] Implement `VulkanThreadState::enqueue_host_func()` (host callback execution after timeline semaphore wait).
- [x] Author precompiled GLSL compute utility kernels in `resources/vulkan_kernels.glsl`, compiled offline via `glslangValidator --target-env vulkan1.3` into SPIR-V 1.6 and LZ4-packed via `resources/pack.c` into `resources/vulkan_kernels.lz4` / `resources/vulkan_kernels.h` (embedded into `libdrjit-core` via `drjit_embed_binary` and lazily instantiated as `VkPipeline`s on demand).
- [x] Implement `VulkanThreadState::block_reduce()`, `block_prefix_reduce()`, `all()`, `any()`, `all_async()`, and `any_async()` using `Workgroup` shared memory and subgroup reductions/scans (`subgroupAdd`/`Mul`/`Min`/`Max`/`Or`/`And`, `subgroupExclusive*`).
- [x] Implement `VulkanThreadState::reduce_dot()` (two-stage chunked dot-product reduction with `Float64` host-sum finalization and `Float16`/`Float32` GPU block reduction).
- [x] Implement `VulkanThreadState::compress()` (exclusive `UInt8 -> UInt32` prefix scan + `compress_scatter` compaction kernel).
- [x] Implement `VulkanThreadState::block_mkperm()` (both the stable `tiny` per-subgroup shared-memory histogram path using `mkperm_match_any` subgroup ballot matching and the global-atomic fallback path, plus `mkperm_detect_offsets` for `OptimizeCalls=1`).

---

## Milestone 4: Ray Tracing Queries (`VK_KHR_acceleration_structure` & `VK_KHR_ray_query`)

**Goal**: Wrap caller-built GPU BLAS/TLAS acceleration structures and execute inline SPIR-V ray tracing queries (`SPV_KHR_ray_query`) for both triangle meshes and custom procedural AABB geometries (`jit_isect_*`).

> Done in commit `1b858db`. Verified by `tests/vulkan_triangle.cpp` (`vulkan_triangle` passing all 5 test suites on NVIDIA RTX A6000 with 0 leaks: single-triangle BLAS/TLAS construction & 16x16 closest-hit ray grid with kernel caching & scene cleanup callbacks, shadow rays / active lane masks / per-lane visibility masks / backface culling, custom procedural AABB sphere intersection & hybrid triangle+AABB scenes, `Float64` ray tracing & custom intersection, and frozen function scene rebinding via `jit_freeze_*`).

### 4.1 Public Ray Tracing API & Scene Lifecycle (`include/drjit-core/vulkan.h`, `include/drjit-core/jit.h`, `src/vulkan_core.cpp`, `src/isect.cpp`)
> **Design Choice (Lean Caller-Owned TLAS API Mirroring `metal.h` & `optix.h`)**: Rather than adding high-level `VulkanGeometryDesc` / `VulkanBLASDesc` / `VulkanInstanceDesc` structs and building acceleration structures inside `drjit-core`, `include/drjit-core/vulkan.h` follows the lean caller-owned TLAS design of `include/drjit-core/metal.h` and `include/drjit-core/optix.h`. Callers build BLAS/TLAS handles using the Vulkan API on `jit_malloc` buffers (which automatically carry `VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR` when `dev.supports_accel` is true) and record builds onto Dr.Jit's stream via `jit_vulkan_command_buffer()`. Unlike `jit_metal_configure_scene`, no `void **resources` array is required because Vulkan `PhysicalStorageBuffer64` allocations and acceleration structures are resident by default once bound to `VkDeviceMemory`.

- [x] Create `ext/drjit-core/include/drjit-core/vulkan.h` declaring `jit_vulkan_configure_scene()`, `jit_vulkan_ray_trace()`, `jit_vulkan_scene_owner_handle()`, `jit_vulkan_scene_set_cleanup()`, and `jit_vulkan_command_buffer()`, plus `jit_vulkan_supports_ray_tracing()` in `include/drjit-core/jit.h` (with non-Vulkan stubs in `src/api.cpp`).
- [x] Enable `VK_KHR_deferred_host_operations`, `VK_KHR_acceleration_structure`, and `VK_KHR_ray_query` (and `VkPhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure` + `VkPhysicalDeviceRayQueryFeaturesKHR::rayQuery`) during logical device creation in `src/vulkan_core.cpp` when supported by the physical device and loader (`jitc_vulkan_has_accel_api()`).
- [x] Add `VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR` to `vulkan_buffer_new()` in `src/vulkan_core.cpp` whenever `dev.supports_accel` is true so standard `jit_malloc()` allocations can be used directly as vertex/index/AABB/instance build inputs, scratch buffers, and BLAS/TLAS backing storage.
- [x] Implement `VulkanScene` and `jitc_vulkan_configure_scene()` in `src/vulkan.h` and `src/vulkan_core.cpp`:
  - [x] Query the 64-bit TLAS device address via `vkGetAccelerationStructureDeviceAddressKHR`.
  - [x] Allocate a host-visible shared `isect_table` buffer (`uint64_t[n_isect_entries]`) when `n_isect_entries > 0`, storing each bound `JitIsectBinding`'s 64-bit closure data address (`b->record.data`) and passing `isect_table_address` as a `ResourceKind::IFT` kernel parameter.
  - [x] Create the reference-counted `owner_var` (`VarType::Pointer` with a destruction callback that waits on `last_submitted_value`, invokes the user `cleanup_callback` outside `state.lock`, and frees `isect_table`) and `resource_var` (`ResourceKind::Accel`), resolving `ResourceKind::Accel` and `ResourceKind::IFT` via `kernel.param_info[i].kind` in `VulkanThreadState::launch()` (supporting `jit_freeze_replay()`).
- [x] Implement `jitc_vulkan_ray_trace()` in `src/vulkan_core.cpp`, accepting 9 or 10 ray arguments (both `VarType::Float32` and `VarType::Float64`) and scheduling the 8 `VarKind::TraceRay` output variables (`valid`, `t`, `u`, `v`, `inst_id`, `prim_id`, `geom_id`, `user_instance_id`).
- [x] Extend `src/isect.cpp` (`jit_isect_begin`, `jitc_isect_finalize`, `jit_isect_bind`, `jit_isect_unbind`, `jitc_isect_hash_scene`, `jitc_isect_DeferredData_upload`, `jitc_isect_assemble_scene`) and `src/eval.cpp` for `JitBackend::Vulkan`.

### 4.2 Inline SPIR-V Ray Query Codegen (`SPV_KHR_ray_query` in `src/vulkan_eval.cpp`)
> **Limitation (Custom Procedural AABB Intersection in `SPV_KHR_ray_query`)**: Inline ray queries in a `GLCompute` shader do not use a hardware Shader Binding Table (SBT) or `MTLIntersectionFunctionTable`, nor does `SPV_KHR_ray_query` provide built-in payload registers for procedural AABB hit attributes (`OpRayQueryGetIntersectionBarycentricsKHR` is valid only for `CommittedIntersectionTriangleKHR`). Consequently:
> 1. All custom intersection functions bound to a scene (`VulkanScene::isect_entries`) are emitted into the entry kernel's `spv_functions` section (`%isect_<hash>` returning `%Tuple_isect_ret = OpTypeStruct %bool %f32 %u32 %u32`) and dispatched inside the `while (OpRayQueryProceedKHR(%rq))` candidate loop via `OpSwitch` on `%rec_idx = OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR + OpRayQueryGetIntersectionGeometryIndexKHR`, loading `%data_ptr = isect_table[%rec_idx]`.
> 2. Custom hit attributes (`attr0`, `attr1`) are tracked in function-local `OpVariable` slots (`%rq_attr0`, `%rq_attr1`) updated whenever `hit && (t_cand >= t_min) && (t_cand <= t_comm)` immediately before `OpRayQueryGenerateIntersectionKHR %rq %t_cand`.
> 3. Because the `rec_idx -> %isect_<hash>` mapping is hashed into the `TraceRay` IR node (`jitc_isect_hash_scene`), rebinding a *different* intersection function to a scene slot triggers a kernel recompile, whereas updating only captured closure data buffers updates `isect_table[entry]` in place without recompiling.

- [x] Emit `OpCapability RayQueryKHR` and `OpExtension "SPV_KHR_ray_query"` when a kernel contains `VarKind::TraceRay`.
- [x] Declare `%accel_t = OpTypeAccelerationStructureKHR`, `%ray_query_t = OpTypeRayQueryKHR`, and function-local `%rq_r<id> = OpVariable %ptr_fn_ray_query Function` in `spv_func_locals`.
- [x] Convert the 64-bit TLAS device address (`accel_h`) via `OpConvertUToAccelerationStructureKHR`.
- [x] Emit `OpRayQueryInitializeKHR %rq_r<id> %tlas %ray_flags %cull_mask %origin %tmin %direction %tmax` (converting `Float64` ray origins, directions, and `tmin`/`tmax` to `%f32` at the hardware ray-query boundary):
  - [x] Fast path for triangle-only scenes (`!has_bbox`): Pass `RayFlagsOpaqueKHR` (`0x1`, plus `RayFlagsTerminateOnFirstHitKHR` `0x4` for shadow rays and `RayFlagsCullBackFacingTrianglesKHR` `0x10` when `geometry_types_mask & 0x8`) and emit a single `OpRayQueryProceedKHR`.
  - [x] Candidate loop for scenes with procedural AABB geometries (`has_bbox`):
    - [x] Emit `while (OpRayQueryProceedKHR(%rq_r<id>))` structured loop (`OpLoopMerge`).
    - [x] Check `OpRayQueryGetIntersectionTypeKHR %u32 %rq_r<id> %c_u32_0 == CandidateIntersectionAABBKHR (1)` (or confirm candidate triangles via `OpRayQueryConfirmIntersectionKHR` in hybrid scenes).
    - [x] Extract candidate `prim_id`, `inst_sbt_offset`, `geom_index`, object-space ray `origin` & `direction`, and current `t_comm = OpRayQueryGetIntersectionTKHR(%rq_r<id>, 1)`.
    - [x] Load `%data_ptr = isect_table[%rec_idx]` and dispatch via `OpSwitch %rec_idx` to `%isect_<hash>(...)` compiled by `jitc_vulkan_assemble_isect()`.
    - [x] On valid hit (`hit && t_cand >= t_min && t_cand <= t_comm`), store custom attributes (`attr0`, `attr1`) to `%rq_attr0` / `%rq_attr1` and invoke `OpRayQueryGenerateIntersectionKHR %rq_r<id> %t_cand`.
- [x] Extract committed hit results after traversal (`OpRayQueryGetIntersectionTypeKHR %u32 %rq_r<id> %c_u32_1 != CommittedIntersectionNoneKHR (0)`):
  - [x] Populate the 8 `TraceRay` output registers: `hit` (`bool`), `t` (`OpRayQueryGetIntersectionTKHR`, upcast to `%f64` for `Float64` rays), `u` & `v` (`OpRayQueryGetIntersectionBarycentricsKHR` for triangles or `OpBitcast %f32` of `%rq_attr0`/`%rq_attr1` for AABBs), `instance_id` (`OpRayQueryGetIntersectionInstanceIdKHR`), `prim_id` (`OpRayQueryGetIntersectionPrimitiveIndexKHR`), `geom_id` (`OpRayQueryGetIntersectionGeometryIndexKHR`), and `user_instance_id` (`OpRayQueryGetIntersectionInstanceCustomIndexKHR`).
- [x] Create `ext/drjit-core/tests/vulkan_triangle.cpp` verifying triangle hits/misses, kernel caching, scene cleanup callbacks, shadow rays, active lane masks, visibility masks, backface culling, custom procedural AABB & hybrid scenes, `Float64` ray tracing & custom intersection, and frozen function scene rebinding.

---

## Milestone 5: Textures, Cooperative Vectors & Top-Level `drjit` C++/Python Bindings

**Goal**: Complete texture support in `drjit-core` and expose `drjit::VulkanArray<T>` in C++ and `drjit.vulkan` / `drjit.vulkan.ad` in Python.

### 5.1 Vulkan Texture Subsystem (`src/vulkan_tex.h`, `src/vulkan_tex.cpp`, `src/vulkan_core.cpp`, `src/vulkan_eval.cpp`)
> Done in commit `96b45ab`. Verified by `test_vulkan_layout` (2,500 static and 6 runtime checks passed), `test_vulkan` (`test20_textures_vulkan` covering 1D/2D/3D textures, `Float32`/`Float16`/`UInt8`/sRGB formats, 1–8 channels, `Nearest`/`Linear` filtering, `Repeat`/`Clamp`/`Mirror` wrap modes, masked & unmasked `jit_tex_lookup`, `jit_tex_lookup_lod`, `jit_tex_lookup_grad`, `jit_tex_bilerp_fetch`, `jit_tex_write` including writable 8-bit sRGB textures, and texture sampling inside `jit_var_call`; 20/20 passed on NVIDIA RTX A6000); 0 regressions across all Vulkan test suites.

- [x] Implement `jitc_vulkan_tex_create`, `jitc_vulkan_tex_wrap`, `jitc_vulkan_tex_map`, `jitc_vulkan_tex_unmap`, `jitc_vulkan_tex_get_shape`, `jitc_vulkan_tex_get_indices`, `jitc_vulkan_tex_memcpy_d2t`, `jitc_vulkan_tex_memcpy_t2d`, `jitc_vulkan_tex_lookup`, `jitc_vulkan_tex_lookup_lod`, `jitc_vulkan_tex_lookup_grad`, `jitc_vulkan_tex_bilerp_fetch`, `jitc_vulkan_tex_write`, `jitc_vulkan_tex_native_handle`, and `jitc_vulkan_tex_destroy` in dedicated `src/vulkan_tex.h` and `src/vulkan_tex.cpp` files using `VkImage`, `VkImageView`, and `VkSampler`.
- [x] Manage a persistent update-after-bind bindless descriptor set (`set = 0`: `binding 0` `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE`, `binding 1` `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE`, `binding 2` `VK_DESCRIPTOR_TYPE_SAMPLER`) bound once per command buffer in `vulkan_stream_ensure_recording()`, with timeline-fenced deferred image destruction (`pending_tex_destroys`) and cached `VkSampler` slots.
- [x] Add 6 GPU compute deinterleave/interleave kernels (`deinterleave_{u8,u16,u32}`, `interleave_{u8,u16,u32}`) to `resources/vulkan_kernels.glsl` and `resources/pack.c` for 3-channel auto-padding and $>4$-channel multi-sub-texture transfers.
- [x] Emit SPIR-V texture operations in `src/vulkan_eval.cpp` (`jitc_vulkan_render_tex_read` and `jitc_vulkan_render_tex_write`) for `VarKind::TexLookup` (`OpImageSampleExplicitLod ... Lod %c_zero_f32`), `VarKind::TexLookupLod` (`OpImageSampleExplicitLod ... Lod`), `VarKind::TexLookupGrad` (`OpImageSampleExplicitLod ... Grad`), `VarKind::TexFetchBilerp` (`OpImageGather`), and `VarKind::TexWrite` (`OpImageWrite`, including shader linear-to-sRGB conversion for writable 8-bit sRGB textures).

### 5.2 Top-Level `drjit` C++ & Python Bindings (`drjit::VulkanArray`, `drjit.vulkan`, `drjit.vulkan.ad`)
> Done in commits `64e39ee` & `041e680` (`ext/drjit-core`) and `e6b6135a` (`drjit`). Verified by the full `drjit-core` C++ test suite (13/13 binaries) and the full `pytest` suite (`10040 passed`, 0 failures).

- [x] Add `DRJIT_ENABLE_VULKAN` option in top-level `CMakeLists.txt`.
- [x] Widen `ArrayMeta::backend` from 2 to 3 bits in `include/drjit/python.h` (the struct is widened to 64 bits) so that `JitBackend::Vulkan == 4` can be represented.
- [x] Define `VulkanArray<Value_>` (`JitArray<JitBackend::Vulkan, Value_>`) in `include/drjit/jit.h` (and `array_traits.h`, `array_base.h`, `tensor.h`, `texture.h`).
- [x] Add `JitBackend::Vulkan` explicit template instantiations in `include/drjit/autodiff.h`, `src/extra/math.cpp`, and `src/extra/resample.cpp`.
- [x] Create `src/python/vulkan.h`, `src/python/vulkan.cpp`, and `src/python/vulkan_ad.cpp` (modeled on `src/python/metal.*`) and register `JitBackend::Vulkan` in `src/python/main.cpp`.
- [x] Create `drjit/vulkan/__init__.py` and `drjit/vulkan/ad.py`.
- [x] Run and pass the `pytest` test suite (`tests/test_*.py`) on `drjit.vulkan` and `drjit.vulkan.ad` (including 16-bit `scatter_reduce`, IEEE-754 `isnan` bitmask check, `UInt8` texture write rounding, and asynchronous `VkQueryPool` hardware timestamp pooling in `drjit-core`).

---

## Edge Cases & Verification Checklist

Items and cross-backend architectural edge cases to verify across milestone implementations:

1. **Multi-Backend Kernel Cache & Lifetime Management (`state.kernel_cache` & `jitc_kernel_free`)**:
   - *Problem*: `state.kernel_cache` is a single global robin map keyed by `KernelKey{device, flags, hash}`. On Linux/Windows with both `DRJIT_ENABLE_CUDA` and `DRJIT_ENABLE_VULKAN` enabled, both backends use `device_id >= 0` (e.g. `device_id == 0`). Historically, `jitc_kernel_free(int device_id, const Kernel &kernel)` inferred the backend purely from `device_id == -1` (LLVM) vs. `device_id >= 0` (CUDA/Metal), which becomes ambiguous when CUDA and Vulkan coexist in the same process.
   - *Current Solution*: Added `JitBackend backend` to `struct Kernel` in `src/io.h`, populated in `jitc_run()` (`kernel.backend = ts->backend`), allowing `jitc_kernel_free()` to cleanly dispatch to `jitc_vulkan_kernel_free()`, `cuModuleUnload()`, or `free(kernel.llvm.reloc)`.
   - *Items to Check*:
     - [ ] **`KernelKey` Collision Safety**: Check if `KernelKey` should also encode `backend` (or pack it into the unused upper bits of `KernelKey::flags`) so that if a CUDA kernel and a Vulkan kernel ever share identical source hash, device index, and flags, their entries in `state.kernel_cache` remain isolated.
     - [ ] **Dual-Backend Runtime Stress Test**: Verify concurrent or alternating execution and cache eviction (`jitc_flush_kernel_cache()`, `jit_shutdown()`) with both `JitBackend::CUDA` and `JitBackend::Vulkan` initialized in the same binary.
     - [ ] **Memory Footprint Evaluation**: Verify whether keeping `Kernel::backend` vs. encoding `backend` in `KernelKey` is the preferred long-term design.

2. **C99 Signed Modulo vs. SPIR-V / GLSL `OpSRem`**:
   - *Problem*: GLSL 4.50 and SPIR-V `Shader` specify that `OpSRem` produces undefined results if either operand is negative (often lowered by GPU drivers like NVIDIA to unsigned modulo).
   - *Solution*: Lowered signed `VarKind::Mod` as `a - (a / b) * b` using `OpSDiv`, `OpIMul`, and `OpISub`.
   - *Check*: [x] Verify signed negative modulo behavior matches CPU C99 semantics (`-7 % 3 == -1`, `7 % -3 == 1`, `-7 % -3 == -1`).

3. **`Float16` Math Intrinsic Precision (`Sqrt`, `Div`, `Rcp`)**:
   - *Problem*: Vulkan drivers may provide approximate or reduced-precision `OpExtInst %glsl Sqrt` or `OpFDiv` on 16-bit floats that diverge from CPU IEEE-754 constant folding (`drjit::half`).
   - *Solution*: Promoted `Float16` square root and division through `%f32` (`OpFConvert %f32` -> op -> `OpFConvert %f16`).
   - *Check*: [x] Verify precision matching CPU float16 reference values across all operands.

4. **Push-Constant Alignment & Size Limits**:
   - *Problem*: Vulkan mandates `vkCmdPushConstants` offset and size to be multiples of 4 bytes, with a minimum device guarantee of only 128 bytes (`maxPushConstantsSize`).
   - *Solution*: Padded single-parameter launches (8 bytes) to 16 bytes for alignment; automatically fallback to physical storage buffer staging buffer parameter blocks when `n_params * 8 > max_push_constants_size`.
   - *Check*: [x] Verify both push-constant path and storage-buffer fallback path in `21_eval_arithmetic_and_casts` and `18_spirv_assemble`.

5. **Limitation — Monolithic Callable Compilation in Vulkan Compute (`VK_SHADER_STAGE_COMPUTE_BIT`)**:
   - *Limitation*: Unlike LLVM (ORCv2 per-unit object compilation), CUDA (`cuLinkAddData` + indirect function-pointer table), and Metal (`MTLLinkedFunctions` + `MTLVisibleFunctionTable`), standard Vulkan compute pipelines (`VkComputePipelineCreateInfo`) accept only a single `VkShaderModule`, forbid `OpCapability Linkage` (`VUID-RuntimeSpirv-None-06275`), and do not support function pointers in compute shaders.
   - *Consequence*: Indirect callables cannot be compiled or cached as independent driver-level units (`UnitBuilder`s) across kernels. Instead, all callable `OpFunction` definitions referenced by a kernel are emitted into the entry unit's SPIR-V module (`spv_functions`), dispatched via `OpSwitch` over direct `OpFunctionCall`s, and compiled together in a single `vkCreateComputePipelines` invocation.

6. **Design Choices & Limitations — Inline Ray Tracing (`VK_KHR_acceleration_structure` & `VK_KHR_ray_query`)**:
   - *Caller-Owned TLAS vs. Managed Scene Builder*: Consistent with `metal.h` and `optix.h`, `include/drjit-core/vulkan.h` wraps a caller-built `VkAccelerationStructureKHR` TLAS handle (`jit_vulkan_configure_scene(tlas, n_isect_entries, geometry_types_mask)`) rather than exposing Vulkan-specific BLAS/TLAS geometry descriptors in `drjit-core`. Callers record AS builds directly onto Dr.Jit's stream via `jit_vulkan_command_buffer()` using standard `jit_malloc()` buffers (which include `VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR` when `dev.supports_accel` is true).
   - *Monolithic `OpSwitch` Dispatch for Custom Procedural Intersections (`jit_isect_*`)*: Because inline `SPV_KHR_ray_query` in a compute shader has neither a hardware Shader Binding Table (SBT) nor `MTLIntersectionFunctionTable`, custom procedural AABB intersections are emitted as `OpFunction`s into the kernel's SPIR-V module and dispatched via `OpSwitch` on `instanceShaderBindingTableRecordOffset + geometryIndex` inside the `OpRayQueryProceedKHR` loop, reading per-entry closure data pointers from `VulkanScene::isect_table` (`ResourceKind::IFT`). Rebinding a different intersection function hash to a scene slot triggers kernel recompilation, while updating captured closure data buffers updates `isect_table` in place.
   - *Software Attribute Tracking for Procedural Hits*: `OpRayQueryGetIntersectionBarycentricsKHR` is valid only for hardware triangle intersections (`CommittedIntersectionTriangleKHR`). Custom AABB attributes (`attr0`, `attr1`) returned by `%isect_<hash>` are stored in function-local `OpVariable` slots whenever a candidate AABB hit is accepted (`t_min <= t_cand <= t_comm`) before calling `OpRayQueryGenerateIntersectionKHR`.
   - *Hardware 32-bit Float Ray Traversal & Motion Blur*: `SPV_KHR_ray_query` operates strictly in 32-bit `float` (`%f32`) and standard `VK_KHR_ray_query` / `VK_KHR_acceleration_structure` does not accept a hardware motion-blur `time` parameter in `OpRayQueryInitializeKHR` or support cross-vendor motion-blur instance transforms (`VK_NV_ray_tracing_motion_blur` is vendor-specific and requires `OpRayQueryInitializeMotionNV`). `Float64` rays are supported by converting `f64 <-> f32` at the hardware ray-query boundary while running `Float64` custom intersection bodies in double precision, and the ray `time` parameter is forwarded directly to custom procedural intersection functions. In Mitsuba 3 (`VulkanAccel`), scenes with animated instance transforms (`AnimatedTransform` / `sd.has_motion == true`) are rejected at acceleration structure build time with a descriptive runtime error.
   - *No Hardware Curve Primitives (`BSplineCurve` & `LinearCurve`)*: Unlike Embree, OptiX (`OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE` / `OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR`), and Metal (`MTLAccelerationStructureCurveGeometryDescriptor`), standard Vulkan `VK_KHR_acceleration_structure` (`VkGeometryTypeKHR`) only supports `VK_GEOMETRY_TYPE_TRIANGLES_KHR` and `VK_GEOMETRY_TYPE_AABBS_KHR` and provides no native curve geometry type. In Mitsuba 3 (`VulkanAccel`), curve shapes (`bsplinecurve` and `linearcurve`) are rejected at acceleration structure build time with a descriptive runtime error.

7. **Design Choices & Limitations — Texture Subsystem (`src/vulkan_tex.cpp` & `src/vulkan_eval.cpp`)**:
   - *1D Texture Promotion to 2D*: Like Metal (`src/metal_tex.mm`), 1D textures (`ndim == 1`) are allocated as `VK_IMAGE_TYPE_2D` with `height = 1` (sampled at `(u, 0.5f)` and written at integer `(x, 0)`), avoiding a separate 1D descriptor binding and supporting 1D gradient sampling via 2D `Grad`.
   - *Explicit LOD 0.0 for Compute `TexLookup`*: Because `OpImageSampleImplicitLod` is illegal in a `GLCompute` execution model (`VUID-StandaloneSpirv-OpImageSampleImplicitLod-04656`), plain `VarKind::TexLookup` emits `OpImageSampleExplicitLod ... Lod %c_zero_f32` (sampling MIP level 0.0, matching CUDA `tex` and Metal `sample()` in compute shaders).
   - *1/2-Channel sRGB Auto-Padding & Writable sRGB Textures*: When the Vulkan driver does not support `VK_FORMAT_R8_SRGB` or `VK_FORMAT_R8G8_SRGB` (e.g. NVIDIA Linux driver), 1- and 2-channel sRGB sub-textures are automatically padded to 4 storage channels (`VK_FORMAT_R8G8B8A8_SRGB`) using the GPU deinterleave/interleave compute kernels. For writable 8-bit sRGB textures (`writable && srgb`), because Vulkan forbids `VK_IMAGE_USAGE_STORAGE_BIT` on `*_SRGB` formats, the `VkImage` is created with `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` in `*_UNORM` format, `binding 0` (`SAMPLED_IMAGE`) binds an `*_SRGB` `VkImageView` restricted via `VkImageViewUsageCreateInfo` to `VK_IMAGE_USAGE_SAMPLED_BIT`, `binding 1` (`STORAGE_IMAGE`) binds a `*_UNORM` `VkImageView`, and `jitc_vulkan_render_tex_write()` applies linear-to-sRGB conversion on channels `0..2` before `OpImageWrite`.

8. **Mitsuba 3 Integration Limitations (`VulkanAccel` in `src/render/vulkan_accel.cpp`)**:
   - *No Curve Shapes (`bsplinecurve`, `linearcurve`)*: Standard `VK_KHR_acceleration_structure` only supports triangle (`VK_GEOMETRY_TYPE_TRIANGLES_KHR`) and procedural AABB (`VK_GEOMETRY_TYPE_AABBS_KHR`) geometries and lacks a native curve primitive descriptor. Attempting to build a `VulkanAccel` with `ShapeType::BSplineCurve` or `ShapeType::LinearCurve` throws an explicit runtime error.
   - *No Animated Instance Motion Blur (`AnimatedTransform`)*: Standard `VK_KHR_acceleration_structure` and `VK_KHR_ray_query` (`SPV_KHR_ray_query`) do not support hardware time-interpolated instance transforms (unlike OptiX `OptixSRTMotionTransform` or Metal `MTLAccelerationStructureMotionInstanceDescriptor`). Attempting to build a `VulkanAccel` with animated instances (`sd.has_motion == true`) throws an explicit runtime error.
