#!/usr/bin/env python3
"""
Progressive rendering GUI for Mitsuba using Polyscope
Entry point for the application
"""

import os
import mitsuba as mi
import drjit as dr
import argparse
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import ctypes
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
from controller import ViewController

mi_variant = os.environ.get("MITSUBA_VARIANT", "cuda_ad_rgb")
mi.set_variant(mi_variant)
# dr.set_flag(dr.JitFlag.Debug, True)
# dr.set_flag(dr.JitFlag.KernelFreezing, False)
# dr.set_flag(dr.JitFlag.SpillToSharedMemory, False)
# mi.set_variant('llvm_ad_rgb')


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]

class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", ctypes.c_uint64 * 2),  # oversimplified dtype placeholder
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]

def get_device_pointer(input_array: dr.ArrayBase):
    capsule = input_array.__dlpack__()
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = ctypes.c_void_p
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    dl_managed_tensor_ptr = PyCapsule_GetPointer(capsule, b"dltensor")
    mtensor = ctypes.cast(dl_managed_tensor_ptr, ctypes.POINTER(DLManagedTensor)).contents
    return mtensor.dl_tensor.data

def setup_gpu_interop():
    """
    Register drjit GPU interop functions with Polyscope
    """
    if not mi.variant().startswith('cuda'):
        print("Warning: GPU interop requires CUDA variant")
        return False

    try:
        ps_device_func_dict = {
            "map_resource_and_get_array": lambda handle: dr.cuda.map_graphics_resource_array(handle),
            "unmap_resource": lambda handle: dr.cuda.unmap_graphics_resource(handle),
            "register_gl_buffer": lambda native_id: dr.cuda.register_gl_buffer(native_id),
            "register_gl_image_2d": lambda native_id: dr.cuda.register_gl_texture(native_id),
            "unregister_resource": lambda handle: dr.cuda.unregister_cuda_resource(handle),
            "get_array_ptr": lambda input_array: (get_device_pointer(input_array), None, None, None),  # Nones are optional assert hints
            "memcpy_2d": lambda dst_ptr, src_ptr, width, height: dr.cuda.memcpy_2d_to_array_async(
                dst_ptr, src_ptr, src_pitch=width, height=height, from_host=False
            ),
        }
        ps.set_device_interop_funcs(ps_device_func_dict)
        return True

    except Exception as e:
        print(f"Failed to setup GPU interop: {e}")
        return False


@dr.freeze
def render_once(accumulation_buffer, weight, scene, sensor, integrator, spp, seed):
    image = mi.render(scene, spp=spp, sensor=sensor, integrator=integrator, seed=seed)
    return dr.lerp(accumulation_buffer, image, weight)


@dataclass
class BaseGUIState:
    """Base GUI runtime state (auto-scanned for snapshot save/load)"""
    rendering_spp: int = 1
    is_rendering: bool = True
    progressive_rendering: bool = False
    overlay_hint: bool = False
    tonemap_exposure: float = 0.0


class MitsubaGUI:
    def __init__(self, scene_path=None, program_name="Mitsuba"):

        # Setup GPU interop before initializing Polyscope
        self.gpu_interop_enabled = setup_gpu_interop()
        if self.gpu_interop_enabled:
            self.use_gpu = True
        else:
            self.use_gpu = False
            print("GPU interop not available - will use CPU fallback (slower)")

        # GUI state (can be overridden by subclasses before calling super().__init__)
        if not hasattr(self, 'gui_state'):
            self.gui_state = BaseGUIState()

        # Mitsuba rendering related
        self.integrator = None
        self.integrator_params = None
        self.gui_sensor = None
        self.gui_sensor_params = None
        self.accumulation_buffer = None
        self.accumulated_spp = 0
        self.rendering_seed = dr.opaque(mi.UInt32, 0)  # Counter for seed
        self.render_resolution = [1920, 1080]
        self.max_accumulated_spp = 8192
        self.use_denoiser = False
        self.denoiser = None

        # GUI related
        self.gui_visible = True
        self.benchmark = False

        # Polyscope image display
        self.main_render_buffer = None
        self.mitsuba_render_buffer = dr.ones(
            mi.Float,
            self.render_resolution[0] * self.render_resolution[1] * 4
        )  # RGBA, FP32


        # Camera state tracking
        self.last_camera_params = None
        self.movement_speed = 0.005  # For WASD movement

        # GPU-based camera controller (will be initialized after scene loading)
        self.controller = None
        self.last_controller_generation = -1

        # Screenshot configuration
        self.screenshot_path = "image.exr"

        # Initialize Polyscope
        ps.set_program_name(program_name)
        ps.set_up_dir("y_up")
        ps.set_front_dir("neg_z_front")
        ps.look_at((0, 0, 1), (0, 0, 0))
        ps.set_navigation_style("first_person")
        ps.set_build_gui(False)  # No default GUI
        ps.set_enable_vsync(False)
        ps.set_max_fps(1000)
        ps.set_background_color((1, 1, 1))
        ps.set_window_size(1920, 1080)
        ps.init()

        # Load scene
        self.scene_path = scene_path
        self.scene, self.integrator = self.load_scene(scene_path)
        self.scene_params = mi.traverse(self.scene)
        self.integrator_params = mi.traverse(self.integrator)
        self.bbox = self.scene.bbox()

        # Extension point for derived classes to setup additional functionality
        self.custom_init_setup()

        # Initialize sensor from scene or create default
        self.gui_sensor = self.create_sensor()
        self.gui_sensor_params = mi.traverse(self.gui_sensor)

        # Update sensor to_world matrix and FOV to match scene camera if available
        if self.scene.sensors():
            scene_sensor_params = mi.traverse(self.scene.sensors()[0])
            self.gui_sensor_params['to_world'] = mi.AffineTransform4f(scene_sensor_params['to_world'])
            if 'x_fov' in scene_sensor_params:
                self.gui_sensor_params['x_fov'] = scene_sensor_params['x_fov']
        # For film size, ignore scene camera but match polyscope window size
        self.gui_sensor_params['film.size'] = [self.render_resolution[0], self.render_resolution[1]]
        self.gui_sensor_params.update()

        # Initialize GPU-based camera controller
        cam_matrix = mi.ScalarMatrix4f(dr.slice(self.gui_sensor_params['to_world'].matrix))
        self.controller = ViewController(
            cam_matrix=cam_matrix,
            bbox=self.bbox,
            view_size=max(self.render_resolution[0], self.render_resolution[1])
        )

        # Initialize buffers
        self.reset_renderer()
        self.rendering_seed *= 0

        self.main_render_buffer = self.setup_image_display(
            "Mitsuba Render",
            self.render_resolution[0],
            self.render_resolution[1],
        )

    #####################################################################
    # Functions below are possible extension points for derived classes
    def load_scene(self, path):
        """Load a Mitsuba scene from XML file"""
        scene = mi.load_file(str(path), optimize=True)
        integrator = self.create_integrator("prb")
        return scene, integrator

    def custom_init_setup(self):
        """Extension point for derived classes to initialize custom components (optimizer, scaler, etc.)."""
        pass

    def custom_gui_callback_step(self):
        """Extension point for derived classes to perform custom processing"""
        pass

    def draw_custom_gui(self, reset_rendering):
        """Extension point for derived classes to draw custom GUI elements"""
        return False  # Return True if reset_rendering should be triggered

    def handle_custom_input(self):
        """Extension point for derived classes to handle custom input"""
        pass

    def add_custom_hints(self, hints):
        """Extension point for derived classes to add custom hints"""
        pass
    #####################################################################

    def reset_renderer(self):
        """Reset accumulation buffers"""
        self.accumulation_buffer = dr.opaque(
            mi.TensorXf,
            0,
            shape=(self.render_resolution[1], self.render_resolution[0], 3)
        )
        self.accumulated_spp = 0

    def create_sensor(self):
        """Create a sensor with the current settings"""
        return mi.load_dict({
            'type': 'perspective',
            'film': {
                'type': 'hdrfilm',
                'width': self.render_resolution[0],
                'height': self.render_resolution[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
                'filter': {'type': 'box'}
            }
        })

    def create_integrator(self, integrator_type="prb"):
        """Create an integrator of the specified type"""
        config = {'type': integrator_type, "max_depth": 8, 'rr_depth': 100}
        return mi.load_dict(config)

    def setup_image_display(self, name, width, height, show_in_imgui_window=False, struct_ref=None):
        try:
            # Create initial dummy image for the quantity
            # NOTE: there seems no easy hack to use uint8 or only 3 channels.
            dummy_image = np.ones((height, width, 4), dtype=np.float32)

            # Add color+alpha image quantity - this creates the texture internally
            ps.add_color_alpha_image_quantity(
                name,
                dummy_image,
                enabled=True,
                image_origin="upper_left",
                show_fullscreen=not show_in_imgui_window,
                show_in_imgui_window=show_in_imgui_window,
                struct_ref=struct_ref
            )

            # Get the buffer for GPU updates
            if struct_ref is not None:
                return struct_ref.get_quantity_buffer(name, "colors")
            else:
                return ps.get_quantity_buffer(name, "colors")

        except Exception as e:
            raise RuntimeError(
                f"Failed to setup image display: {e}"
            ) from e

    def setup_denoiser(self):
        self.denoiser = mi.OptixDenoiser(
            input_size=[self.render_resolution[0], self.render_resolution[1]],
            albedo=False,
            normals=False,
            temporal=False
        )

    def update_buffer(self, buffer, new_image):
        if self.gpu_interop_enabled:
            try:
                buffer.update_data_from_device(new_image)
            except Exception as e:
                self.gpu_interop_enabled = False
                print(f"\033[91mGPU interop failed, falling back to CPU path: {e}\033[0m")
        else:
            # Fallback to CPU
            rgba_flat_np = new_image.numpy()
            rgba_flat_np = np.reshape(rgba_flat_np, (-1, 4))

            buffer.update_data_from_host(rgba_flat_np)

    def gui_callback(self):
        """Main GUI callback for Polyscope"""

        # Extension point for derived classes to perform additional steps
        self.custom_gui_callback_step()

        # Display GUI
        if self.benchmark:
            io = psim.GetIO()
            print(f"FPS: {io.Framerate:.1f} ({1000.0/io.Framerate:.2f} ms)  ({self.render_resolution[0]}x{self.render_resolution[1]})")
        elif self.gui_visible:
            self.draw_gui()

        # Handle camera input and updates
        if not self.benchmark:
            self.handle_input()
            # Update GPU controller and check for camera changes
            if self.controller:
                io = psim.GetIO()
                dt = 1.0 / max(io.Framerate, 1.0)  # Avoid division by zero
                self.controller.update(dt)

                # Check if camera moved and reset renderer if needed
                current_generation = self.controller.generation
                if current_generation != self.last_controller_generation:
                    # Update sensor with controller's camera matrix
                    sensor_mat = mi.AffineTransform4f(self.controller.matrix())
                    dr.make_opaque(sensor_mat)
                    self.gui_sensor_params['to_world'] = sensor_mat
                    self.gui_sensor_params.update()

                    self.reset_renderer()
                    self.last_controller_generation = current_generation

        # Render and display
        if (self.scene and self.gui_state.is_rendering and
            (not self.gui_state.progressive_rendering or self.accumulated_spp < self.max_accumulated_spp)
        ):
            with dr.suspend_grad():
                # Perform one rendering pass

                rendering_spp = self.gui_state.rendering_spp
                if not self.gui_state.progressive_rendering:
                    weight = 1
                else:
                    old_spp = self.accumulated_spp
                    weight = 1 if old_spp == 0 else rendering_spp / (old_spp + rendering_spp)
                    self.accumulated_spp += rendering_spp

                self.accumulation_buffer = render_once(
                    self.accumulation_buffer,
                    dr.opaque(mi.Float, weight),
                    self.scene, self.gui_sensor, self.integrator, rendering_spp,
                    seed=self.rendering_seed
                )

                self.rendering_seed += 1
                dr.make_opaque(self.accumulation_buffer)

                # Update main render display
                if not self.benchmark:
                    self.prepare_mitsuba_render_buffer(self.accumulation_buffer)
                    self.update_buffer(self.main_render_buffer, self.mitsuba_render_buffer)

    def prepare_mitsuba_render_buffer(self, mitsuba_image: mi.TensorXf):

        if self.use_gpu and self.use_denoiser and self.denoiser is not None:
            mitsuba_image = self.denoiser(mitsuba_image)

        if True:
            mitsuba_image = mitsuba_image * (2.0 ** self.gui_state.tonemap_exposure)  # Exposure adjustment
            # Simple ACES tonemapping
            mitsuba_image = dr.clip(mitsuba_image / (1 + mitsuba_image), 0, 1) ** (1/2.2)
            # Reorganize to RGBA  (polyscope limitation)
            num_pixels = mitsuba_image.shape[0] * mitsuba_image.shape[1]
            index = dr.repeat(dr.arange(mi.UInt32, num_pixels) * 4, 3) + dr.tile(mi.UInt32([0, 1, 2]), num_pixels)
            dr.scatter(self.mitsuba_render_buffer, mitsuba_image.array, index)
        else:
            # CPU fallback
            mitsuba_image = mi.Bitmap(mitsuba_image).convert(
                pixel_format=mi.Bitmap.PixelFormat.RGBA,
                component_format=mi.Struct.Type.Float32,
                srgb_gamma=True
            )
            self.mitsuba_render_buffer = dr.ravel(mi.TensorXf(mitsuba_image))

    def save_screenshot(self):
        screenshot_path = Path(self.screenshot_path).expanduser()
        if screenshot_path.parent and not screenshot_path.parent.exists():
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mi.Bitmap(self.accumulation_buffer).write(str(screenshot_path))
            print(f"Saved {screenshot_path}")
        except Exception as e:
            print(f"\033[91mFailed to save image: {e}\033[0m")

    def draw_gui(self):
        # Opaque dark background
        psim.PushStyleColor(psim.ImGuiCol_WindowBg, [0.1, 0.1, 0.1, 1.0])

        def button_green(label, callback):
            psim.PushStyleColor(psim.ImGuiCol_Button, (0.20, 0.32, 0.25, 0.6))
            psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, (0.25, 0.40, 0.31, 0.9))
            psim.PushStyleColor(psim.ImGuiCol_ButtonActive, (0.18, 0.29, 0.23, 1.0))
            if psim.Button(label):
                callback()
            psim.PopStyleColor(3)

        # Position main window at left with fixed width
        io = psim.GetIO()
        psim.SetNextWindowPos([0, 0])
        psim.SetNextWindowSize([400, 0])  # Fixed width 400px, auto height

        flags = psim.ImGuiWindowFlags_NoTitleBar | psim.ImGuiWindowFlags_NoScrollbar | psim.ImGuiWindowFlags_NoMove
        if psim.Begin("Mitsuba Controls", True, flags):

            reset_rendering = False

            # Rendering controls
            if psim.CollapsingHeader("Rendering", psim.ImGuiTreeNodeFlags_DefaultOpen):

                io = psim.GetIO()
                psim.Text(f"FPS: {io.Framerate:.1f} ({1000.0/io.Framerate:.2f} ms)  ({self.render_resolution[0]}x{self.render_resolution[1]})")

                changed, self.gui_state.is_rendering = psim.Checkbox("Render", self.gui_state.is_rendering)
                psim.SameLine()
                changed, self.gui_state.progressive_rendering = psim.Checkbox("Progressive", self.gui_state.progressive_rendering)
                if changed:
                    reset_rendering = True

                # Progress rendering status
                if self.gui_state.progressive_rendering:
                    if self.scene:
                        psim.Text(f"Samples: {self.accumulated_spp}/{self.max_accumulated_spp}")
                    changed, self.max_accumulated_spp = psim.SliderInt("Max SPP", self.max_accumulated_spp, 1, 4096)

                current_k = int(np.log2(max(1, self.gui_state.rendering_spp)))
                changed, k = psim.SliderInt("spp/frame",
                                            current_k,
                                            0, 10,  # k ranges from 0 to 10
                                            format=f"{2**current_k}")  # Display 2^k instead of k
                if changed:
                    self.gui_state.rendering_spp = 2 ** k

                if self.gui_state.progressive_rendering:
                    if psim.Button("Reset rendering"):
                        reset_rendering = True

                # Camera controls - TreeNode, default closed
                if psim.TreeNode("Camera"):
                    if self.controller:
                        camera_modes = ["Turntable", "FPS"]
                        current_mode = 1 if self.controller.fps_mode else 0
                        changed, selected_mode = psim.Combo("Camera Mode", current_mode, camera_modes)
                        if changed:
                            self.controller.fps_mode = (selected_mode == 1)
                    changed, self.movement_speed = psim.SliderFloat("Speed", self.movement_speed, 0.0005, 0.02)

                    button_green("Reset Camera", self.reset_camera)
                    psim.TreePop()

                # Miscellaneous
                if psim.TreeNode("Misc"):
                    changed, self.gui_state.overlay_hint = psim.Checkbox("Overlay Hint", self.gui_state.overlay_hint)

                    if self.use_gpu:
                        changed, self.use_denoiser = psim.Checkbox("Use Optix Denoiser", self.use_denoiser)
                        if changed and self.use_denoiser and self.denoiser is None:
                            self.setup_denoiser()

                    # Exposure slider
                    changed, self.gui_state.tonemap_exposure = psim.SliderFloat(
                        "Exposure",
                        self.gui_state.tonemap_exposure,
                        -5.0,
                        5.0
                    )
                    if psim.IsItemHovered():
                        psim.SetTooltip("Adjust exposure (E/Shift+E)")

                    psim.Separator()
                    psim.Text("Screenshot path:")
                    psim.PushItemWidth(-1)
                    changed, new_path = psim.InputText("##ScreenshotPath", str(self.screenshot_path), 512)
                    psim.PopItemWidth()
                    if changed:
                        self.screenshot_path = new_path or "image.exr"
                    button_green("Save Screenshot", self.save_screenshot)

                    psim.TreePop()

            # Extension point for derived classes to add additional GUI elements
            reset_rendering = self.draw_custom_gui(reset_rendering) or reset_rendering

            if reset_rendering:
                self.reset_renderer()

            psim.End()
        psim.PopStyleColor()

        # Draw hint overlay if needed
        if self.gui_state.overlay_hint:
            # Position overlay in upper-right corner
            w, h = ps.get_window_size()
            overlay_x = w - 330
            overlay_y = 20  # 20px from top

            # Setup overlay window
            psim.SetNextWindowPos([overlay_x, overlay_y])
            psim.SetNextWindowSize([310, 0])  # Fixed width, auto height
            # Window flags for overlay
            overlay_flags = (psim.ImGuiWindowFlags_NoDecoration |
                            psim.ImGuiWindowFlags_AlwaysAutoResize |
                            psim.ImGuiWindowFlags_NoSavedSettings |
                            psim.ImGuiWindowFlags_NoFocusOnAppearing |
                            psim.ImGuiWindowFlags_NoNav |
                            psim.ImGuiWindowFlags_NoMove)
            # Semi-transparent black background
            psim.PushStyleColor(psim.ImGuiCol_WindowBg, [0.0, 0.0, 0.0, 0.7])
            psim.PushStyleVar(psim.ImGuiStyleVar_WindowRounding, 5.0)
            psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, [15.0, 12.0])

            # if psim.Begin("##HintOverlay", True, overlay_flags):
            #     # Make text slightly larger
            #     psim.SetWindowFontScale(1.5)
            #     # Display hint text with color
            #     psim.PushStyleColor(psim.ImGuiCol_Text, [0.9, 0.95, 1.0, 1.0])

            #     hints = []
            #     if self.gui_state.is_rendering:
            #         hints.append("Path tracing rendering")

            #     # Extension point for derived classes to add custom hints
            #     self.add_custom_hints(hints)

            #     if self.gui_state.progressive_rendering and self.gui_state.is_rendering:
            #         hints.append(f"SPP: {self.accumulated_spp}/{self.max_accumulated_spp}")

            #     if self.use_denoiser:
            #         hints.append("Denoising: ON")

            #     # Add status indicator
            #     if self.gui_state.is_rendering:
            #         if hints:
            #             for i, hint in enumerate(hints):
            #                 psim.Text(hint)
            #     else:
            #         psim.Text("Rendering paused")

            #     psim.PopStyleColor()
            #     psim.SetWindowFontScale(1.0)
            #     psim.End()

            psim.PopStyleVar(2)
            psim.PopStyleColor()

    def reset_camera(self):
        """Reset camera using GPU controller"""
        if self.controller:
            self.controller.reset()
        self.reset_renderer()

    def handle_input(self):
        io = psim.GetIO()

        # Skip keyboard shortcuts if ImGui wants keyboard input (e.g., typing in text boxes)
        if io.WantCaptureKeyboard:
            return

        # System shortcuts
        if psim.IsKeyDown(psim.ImGuiKey_LeftCtrl) and psim.IsKeyPressed(psim.ImGuiKey_Q):
            ps.unshow()
            return

        if psim.IsKeyPressed(psim.ImGuiKey_Tab):
            self.gui_visible = not self.gui_visible

        # Extension point for derived classes to handle additional input
        self.handle_custom_input()

        # Rendering controls
        if psim.IsKeyPressed(psim.ImGuiKey_R) and not psim.IsKeyDown(psim.ImGuiKey_LeftShift):
            self.gui_state.is_rendering = not self.gui_state.is_rendering

        if psim.IsKeyPressed(psim.ImGuiKey_P):
            self.gui_state.progressive_rendering = not self.gui_state.progressive_rendering
            self.reset_renderer()

        # Exposure controls
        if psim.IsKeyPressed(psim.ImGuiKey_E) and not psim.IsKeyDown(psim.ImGuiKey_LeftShift):
            # E without shift - increase exposure
            self.gui_state.tonemap_exposure += 0.5
        elif psim.IsKeyPressed(psim.ImGuiKey_E) and psim.IsKeyDown(psim.ImGuiKey_LeftShift):
            # Shift+E - decrease exposure
            self.gui_state.tonemap_exposure -= 0.5

        # Overlay hint toggle
        if psim.IsKeyPressed(psim.ImGuiKey_H):
            self.gui_state.overlay_hint = not self.gui_state.overlay_hint

        # Handle controller input
        if self.controller:
            self.handle_controller_input()

        # Handle mouse input for camera control
        self.handle_mouse_input()

    def handle_controller_input(self):
        """Handle keyboard input for the GPU controller"""
        # Pass keyboard events to controller
        keys_to_check = [
            psim.ImGuiKey_W, psim.ImGuiKey_A, psim.ImGuiKey_S, psim.ImGuiKey_D,
            psim.ImGuiKey_C, psim.ImGuiKey_Space, psim.ImGuiKey_F, psim.ImGuiKey_M,
            psim.ImGuiKey_R, psim.ImGuiKey_Equal, psim.ImGuiKey_Minus, psim.ImGuiKey_LeftShift,
            psim.ImGuiKey_H
        ]

        for key in keys_to_check:
            if psim.IsKeyPressed(key):
                self.controller.key_event(key, True)
            elif psim.IsKeyReleased(key):
                self.controller.key_event(key, False)

    def handle_mouse_input(self):
        """Handle mouse input for camera control"""
        if not self.controller:
            return

        io = psim.GetIO()

        # Skip camera control if ImGui wants mouse input (e.g., dragging sliders)
        if io.WantCaptureMouse:
            return

        # Handle mouse motion with buttons
        if io.MouseDown[0] or io.MouseDown[1] or io.MouseDown[2]:  # Left, right, middle
            mouse_pos = io.MousePos
            mouse_delta = io.MouseDelta

            pos = type('Position', (), {'x': mouse_pos[0], 'y': mouse_pos[1]})()
            rel = type('Relative', (), {'x': mouse_delta[0], 'y': mouse_delta[1]})()

            button_flags = 0
            if io.MouseDown[0]:  # Left button
                button_flags |= 1
            if io.MouseDown[1]:  # Right button
                button_flags |= 2
            if io.MouseDown[2]:  # Middle button
                button_flags |= 4

            if mouse_delta[0] != 0 or mouse_delta[1] != 0:
                self.controller.mouse_motion(pos, rel, button_flags)

        # Handle scroll wheel
        if io.MouseWheel != 0:
            offset = type('Offset', (), {'x': 0, 'y': io.MouseWheel})()
            self.controller.scroll(offset)

    def run(self):
        """Main run loop"""
        ps.set_user_callback(self.gui_callback)
        ps.show()


def no_opengl_rendering(gui: MitsubaGUI):
    ## Benchmark pure rendering performance without display
    import time

    # Warmup
    for _ in range(10):
        gui.accumulation_buffer = render_once(
            gui.accumulation_buffer,
            1.0,
            gui.scene, gui.gui_sensor, gui.integrator, gui.gui_state.rendering_spp,
            seed=gui.rendering_seed
        )
        gui.rendering_seed += 1
        dr.eval(gui.accumulation_buffer)

    dr.sync_thread()
    num_iters = 5000
    start_time = time.perf_counter()

    for i in range(num_iters):
        gui.accumulation_buffer = render_once(
            gui.accumulation_buffer,
            1.0,
            gui.scene, gui.gui_sensor, gui.integrator, gui.gui_state.rendering_spp,
            seed=gui.rendering_seed
        )
        gui.rendering_seed += 1
        dr.eval(gui.accumulation_buffer)

    dr.sync_thread()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    fps = num_iters / elapsed_time
    ms_per_frame = (elapsed_time / num_iters) * 1000

    print(f"\n{'='*50}")
    print(f"Pure rendering performance (no display):")
    print(f"  Total time: {elapsed_time:.3f} seconds")
    print(f"  FPS: {fps:.1f}")
    print(f"  Time per frame: {ms_per_frame:.3f} ms")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Mitsuba Progressive Rendering GUI')
    parser.add_argument('scene', nargs='?',
                      help='Path to scene XML file')

    args = parser.parse_args()

    gui = MitsubaGUI(args.scene)
    gui.run()
    # no_opengl_rendering(gui)


if __name__ == "__main__":
    main()
