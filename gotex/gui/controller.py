from __future__ import annotations

from drjit.scalar import Array3f, Quaternion4f, Matrix3f, Matrix4f, Array2f, Array3f
import drjit as dr
import mitsuba as mi
import polyscope.imgui as psim


class ViewController:
    # Camera state
    pos:           Array3f      # Camera position
    quat:          Quaternion4f # Orientation quaternion (target)
    pos_init:      Array3f      # State at construction time
    quat_init:     Quaternion4f # (ditto)
    pos_smooth:    Array3f      # Current smoothed interpolant
    quat_smooth:   Quaternion4f # (ditto)
    motion:        float        # Field to track the presence of motion
    focus_dist:    float        # Distance from camera to focal plane

    # Scene state
    bbox: mi.ScalarBoundingBox3f
    scene_scale : float    # Size of the scene (in arbitrary spatial units)

    # GUI state
    inv_view_size: float   # Size of the view in pixels (FOV axis)
    fps_mode : bool        # Switches between turntable/FPS camera mode
    keys: set              # Set of currently pressed keyboard keys (GLFW key codes)
    speed : float          # Speed multiplier for spatial motion
    generation: int        # Generation counter for accumulation restarts

    def __init__(self, cam_matrix: Matrix4f, bbox: mi.ScalarBoundingBox3f, view_size: int):
        self.bbox = bbox
        self.scene_scale = dr.norm(bbox.extents())
        self.inv_view_size = 1.0 / view_size
        self.set_init_matrix(cam_matrix)
        self.keys = set()
        self.up = Array3f(0, 1, 0)
        self.speed = 0.3
        self.smooth_rate = 5
        self.pos_smooth = Array3f(self.pos_init)
        self.quat_smooth = Quaternion4f(self.quat_init)
        self.generation = 0
        self.reset()
        self.motion = 0

    def set_init_matrix(self, cam_matrix: Matrix4f):
        self.quat_init = dr.matrix_to_quat(Matrix3f(cam_matrix))
        self.pos_init = Array3f(cam_matrix[0, 3], cam_matrix[1, 3], cam_matrix[2, 3]) # XXX
        self.focus_dist = dr.norm(self.pos_init - self.bbox.center())

    def set_matrix(self, cam_matrix):
        if not isinstance(cam_matrix, Matrix4f):
            cam_matrix = Matrix4f(dr.slice(cam_matrix))
        quat = dr.matrix_to_quat(Matrix3f(cam_matrix))
        pos = Array3f(cam_matrix[0, 3], cam_matrix[1, 3], cam_matrix[2, 3])

        self.pos = pos
        self.quat = quat
        self.pos_smooth = Array3f(pos)
        self.quat_smooth = Quaternion4f(quat)
        self.mark_dirty()

    def matrix(self):
        matrix = Matrix4f(dr.quat_to_matrix(Matrix4f, self.quat_smooth))
        matrix[0, 3] = self.pos_smooth.x # XXX
        matrix[1, 3] = self.pos_smooth.y
        matrix[2, 3] = self.pos_smooth.z
        return matrix

    def reset(self):
        self.quat = Quaternion4f(self.quat_init)
        self.pos = Array3f(self.pos_init)
        self.fps_mode = True
        self.motion = 1
        self.mark_dirty()

    def mouse_motion(self, pos, rel, button):
        rel = Array2f(rel.x, rel.y) * self.inv_view_size
        left = dr.quat_apply(self.quat, Array3f(1, 0, 0))

        # Left drag: turntable/FPS rotation
        if button & 1:
            rot_sensitivity = 0.35 if self.fps_mode else 1.0
            amount = rel*2.0 * dr.pi * rot_sensitivity

            quat_updt = dr.rotate(Quaternion4f, self.up, -amount.x) * \
                        dr.rotate(Quaternion4f, left, amount.y) * self.quat

            if not self.fps_mode:
                fwd_old = dr.quat_apply(self.quat, Array3f(0, 0, 1))
                fwd_new = dr.quat_apply(quat_updt, Array3f(0, 0, 1))
                self.pos += (fwd_old - fwd_new) * self.focus_dist

            self.quat = quat_updt

        # Right drag - zoom (turntable mode only)
        if button & 2 and not self.fps_mode:
            focus_dist = max(0.001, self.focus_dist + rel.y * self.scene_scale*3)
            fwd = dr.quat_apply(self.quat, Array3f(0, 0, 1))
            self.pos -= fwd * (focus_dist - self.focus_dist)
            self.focus_dist = focus_dist

        # Middle drag: translation
        if button & 4:
            self.pos += (left * rel.x + self.up * rel.y) * self.scene_scale

        if button:
            self.motion = 1

        return button != 0

    def scroll(self, offset):
        fwd = dr.quat_apply(self.quat, Array3f(0, 0, 1))
        if self.fps_mode:
            self.pos += fwd * (offset.y * self.speed * self.scene_scale * 0.1)
        else:
            focus_dist = max(0.001, self.focus_dist - offset.y * self.scene_scale*0.1)
            self.pos -= fwd * (focus_dist - self.focus_dist)
            self.focus_dist = focus_dist
        self.motion = 1
        return True

    def key_event(self, key, pressed):
        if pressed:
            self.keys.add(key)
            if key == psim.ImGuiKey_F:
                self.fps_mode = not self.fps_mode
            elif key == psim.ImGuiKey_M: # Enable/disable smoothing
                self.smooth_rate = float('inf') if self.smooth_rate < float('inf') else 8.0
            elif key == psim.ImGuiKey_R and psim.ImGuiKey_LeftShift not in self.keys: # Reset camera view (only if shift not held)
                # Reset
                self.reset()
            elif key in [psim.ImGuiKey_Equal, psim.ImGuiKey_Minus]:  # Speed
                self.speed *= 1.5 if key == psim.ImGuiKey_Equal else 0.67
        else:
            self.keys.discard(key)
        return True

    def update(self, dt):
        if self.keys:
            mat = dr.quat_to_matrix(Matrix3f, self.quat).T

            move = Array3f(0)
            if psim.ImGuiKey_S in self.keys: move -= mat[2]
            if psim.ImGuiKey_W in self.keys: move += mat[2]
            if psim.ImGuiKey_D in self.keys: move -= mat[0]
            if psim.ImGuiKey_A in self.keys: move += mat[0]
            if psim.ImGuiKey_C in self.keys: move -= mat[1]
            if psim.ImGuiKey_Space in self.keys: move += mat[1]

            if dr.squared_norm(move) > 0:
                self.fps_mode = True
                # Apply 3x speed boost if shift is held
                speed_multiplier = 3.0 if psim.ImGuiKey_LeftShift in self.keys else 1.0
                self.pos += move * self.scene_scale * self.speed * speed_multiplier * dt
                self.motion = 1

        t = 1 - dr.exp(-self.smooth_rate * dt)
        self.quat = dr.normalize(self.quat)
        self.quat_smooth = dr.normalize(dr.slerp(self.quat_smooth, self.quat, t))

        if self.fps_mode:
            self.pos_smooth = dr.lerp(self.pos_smooth, self.pos, t)
        else:
            fwd = Array3f(0, 0, 1)
            fwd_diff = dr.quat_apply(self.quat, fwd)-dr.quat_apply(self.quat_smooth, fwd)
            self.pos_smooth = self.pos + self.focus_dist*fwd_diff

        if self.motion > 1e-3:  # Increment generation while still moving/settling
            self.mark_dirty()
        self.motion = dr.lerp(self.motion, 0, t)

    def mark_dirty(self):
        self.generation = (self.generation + 1) & 0xFFFFFFFF
