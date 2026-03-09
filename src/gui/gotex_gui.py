#!/usr/bin/env python3
"""
GoTex GUI - Interactive texture optimization using Stable Diffusion
Extends the base Mitsuba GUI with real-time optimization capabilities
"""

import sys
import os
import random
import argparse
import traceback
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass

import torch
import drjit as dr
import mitsuba as mi
import polyscope as ps
import polyscope.imgui as psim

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui_base import MitsubaGUI, BaseGUIState
from src.models.sd import StableDiffusion
from models.instaflow import Instaflow
from src.renderer import randomize_sensor, get_depth


@dataclass
class GoTexGUIState(BaseGUIState):
    """Extended GUI state for optimization"""
    # Optimization controls
    is_optimizing: bool = False
    learning_rate: float = 3e-2
    nb_acc_steps: int = 1
    nb_sensors: int = 128
    cfg_scale: float = 7.5
    loss_ema_epsilon: float = 0.9  # EMA smoothing factor
    min_time: float = 0.02
    max_time: float = 0.98
    
    # Optimization progress
    opt_step: int = 0
    current_loss: float = 0.0
    acc_step_counter: int = 0
    
    # Scene configuration
    scene_name: str = "painting"
    is_2d_scene: bool = True


class GoTexGUI(MitsubaGUI):
    """
    Extended Mitsuba GUI with integrated texture optimization using Stable Diffusion.
    """
    
    def __init__(self, scene_path=None, prompt_override=None, program_name="GoTex - Texture Optimization"):
        # Initialize custom GUI state before calling parent constructor
        self.gui_state = GoTexGUIState()
        
        # Optimization components (initialized in custom_init_setup)
        self.sd = None
        self.opt = None
        self.scene_metadata = None
        self.sd_config = None
        self.camera_to_world_key = 'sensor.to_world'
        self.prompt_override = prompt_override
        
        # Torch setup
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Call parent constructor
        super().__init__(scene_path=scene_path, program_name=program_name)
    
    def load_scene(self, path):
        """Load a custom scene with Stable Diffusion configuration"""
        
        # Dynamically import load_scene from the provided path
        if path is None:
            # Use default scene
            from scenes.painting import load_scene
        else:
            scene_path = Path(path)
            if scene_path.suffix == '.py':
                # It's a file path, load as module
                spec = importlib.util.spec_from_file_location("custom_scene", scene_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                load_scene = module.load_scene
            else:
                # Assume it's a module path like "scenes.painting" or "scenes.dragon"
                module = importlib.import_module(path)
                load_scene = module.load_scene
        
        scene, _, scene_metadata, sd_config = load_scene(512)
        
        # Store metadata and config for later use
        self.scene_metadata = scene_metadata
        self.sd_config = sd_config
        self.gui_state.scene_name = scene_metadata['scene_name']
        self.gui_state.is_2d_scene = scene_metadata['is_2d']
        
        # Create integrator (scene loader returns it in the dict, but we need separate)
        integrator = self.create_integrator("path")
        
        return scene, integrator
    
    def custom_init_setup(self):
        """Initialize optimization components"""
        # Set random seeds
        seed = 40
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Override prompt if provided
        if self.prompt_override is not None:
            self.sd_config.prompt = self.prompt_override
            print(f"Using custom prompt: {self.prompt_override}")
        
        # Initialize CFG scale from config
        self.gui_state.cfg_scale = self.sd_config.get('guidance_scale', 7.5)
        self.gui_state.min_time = self.sd_config.get('min_time', 0.02)
        self.gui_state.max_time = self.sd_config.get('max_time', 0.98)
        
        # Initialize Stable Diffusion model
        print("Loading Stable Diffusion model...")
        self.sd = Instaflow(
            config=self.sd_config,
            instaflow=False,
            device=self.torch_device,
            enable_offload=False
        )
        del self.sd_config
        print("Stable Diffusion model loaded!")
        
        # Setup optimizer for scene parameters
        self.scene_params.keep([r'.*\.reflectance\.data', self.camera_to_world_key])
        self.opt = mi.ad.Adam(
            lr=self.gui_state.learning_rate,
            params={k: v for k, v in self.scene_params.items() if self.camera_to_world_key not in k}
        )
        self.scene_params.update(self.opt)
    
    def custom_gui_callback_step(self):
        """Perform optimization step if enabled"""
        if not self.gui_state.is_optimizing:
            return
        
        try:
            # Randomize camera viewpoint for 3D scenes
            if not self.gui_state.is_2d_scene:
                randomize_sensor(
                    self.scene_params,
                    sensor_to_world_key=self.camera_to_world_key,
                    sensor_idx=random.randint(0, self.gui_state.nb_sensors - 1),
                    sensor_count=self.gui_state.nb_sensors,
                    target=self.scene_metadata.get('target', mi.ScalarVector3f(0, 0, 0)),
                    radius=self.scene_metadata.get('radius', 50)
                )
            
            # Render depth and image
            dr_depth = get_depth(self.scene, sensor=self.scene.sensors()[0])
            dr_image = mi.render(self.scene, params=self.scene_params, seed=self.gui_state.opt_step)

            # Simple tone mapping for Stable Diffusion input
            dr_image = dr_image / (dr_image + 1)
            dr_image = dr_image ** (1/2.2)
            
            # Compute loss using Stable Diffusion
            loss_rdfs = self.sd.compute_rdfs_loss(dr_image, dr_depth)
            loss = loss_rdfs / self.gui_state.nb_acc_steps
            
            # Backward pass
            dr.backward(loss)
            
            # Update counter
            self.gui_state.acc_step_counter += 1
            
            # Update loss with exponential moving average
            new_loss = float(loss.item())
            if self.gui_state.opt_step == 0 and self.gui_state.acc_step_counter == 1:
                # Initialize on first step
                self.gui_state.current_loss = new_loss
            else:
                # EMA: loss = old_loss * epsilon + new_loss * (1 - epsilon)
                epsilon = self.gui_state.loss_ema_epsilon
                self.gui_state.current_loss = self.gui_state.current_loss * epsilon + new_loss * (1 - epsilon)
            
            # Optimization step after accumulating gradients
            if self.gui_state.acc_step_counter >= self.gui_state.nb_acc_steps:
                self.opt.step()
                
                # Clip values to valid range
                for k, v in self.opt.items():
                    self.opt[k] = dr.clip(v, 0, 1)
                
                self.scene_params.update(self.opt)
                
                # Reset accumulation counter and increment step
                self.gui_state.acc_step_counter = 0
                self.gui_state.opt_step += 1
                
                # Update accumulation buffer for display
                self.reset_renderer()
            
        except Exception as e:
            print(f"Optimization error: {e}")
            traceback.print_exc()
            self.gui_state.is_optimizing = False
    
    def draw_custom_gui(self, reset_rendering):
        """Draw optimization-specific GUI elements"""
        if psim.CollapsingHeader("Optimization", psim.ImGuiTreeNodeFlags_DefaultOpen):
            
            # Optimization controls
            psim.PushItemWidth(200)
            
            # Start/Stop optimization button
            if self.gui_state.is_optimizing:
                psim.PushStyleColor(psim.ImGuiCol_Button, (0.8, 0.2, 0.2, 0.6))
                psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, (0.9, 0.3, 0.3, 0.9))
                psim.PushStyleColor(psim.ImGuiCol_ButtonActive, (0.7, 0.1, 0.1, 1.0))
                if psim.Button("Stop Optimization (O)"):
                    self.gui_state.is_optimizing = False
                psim.PopStyleColor(3)
            else:
                psim.PushStyleColor(psim.ImGuiCol_Button, (0.20, 0.32, 0.25, 0.6))
                psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, (0.25, 0.40, 0.31, 0.9))
                psim.PushStyleColor(psim.ImGuiCol_ButtonActive, (0.18, 0.29, 0.23, 1.0))
                if psim.Button("Start Optimization (O)"):
                    self.gui_state.is_optimizing = True
                    self.gui_state.opt_step = 0
                    self.gui_state.acc_step_counter = 0
                psim.PopStyleColor(3)
            
            psim.SameLine()
            if psim.Button("Reset Optimizer"):
                self.opt = mi.ad.Adam(
                    lr=self.gui_state.learning_rate,
                    params={k: v for k, v in self.scene_params.items() if self.camera_to_world_key not in k}
                )
                self.scene_params.update(self.opt)
                self.gui_state.opt_step = 0
                self.gui_state.acc_step_counter = 0
                reset_rendering = True
            
            # Optimization progress
            psim.Text(f"Step: {self.gui_state.opt_step}")
            psim.SameLine()
            psim.Text(f"| Loss: {self.gui_state.current_loss:.6f}")
            
            psim.Separator()
            
            # Learning rate
            changed, new_lr = psim.SliderFloat(
                "Learning Rate",
                self.gui_state.learning_rate,
                0.0001, 0.1,
                format="%.4f",
                flags=psim.ImGuiSliderFlags_Logarithmic
            )
            if changed:
                self.gui_state.learning_rate = new_lr
                # Update optimizer learning rate
                if self.opt is not None:
                    for param in self.opt.values():
                        if hasattr(param, 'lr'):
                            param.lr = new_lr
            
            # Accumulation steps
            changed, new_acc = psim.SliderInt(
                "Accumulation Steps",
                self.gui_state.nb_acc_steps,
                1, 10
            )
            if changed:
                self.gui_state.nb_acc_steps = new_acc
            
            # Number of sensors (for 3D scenes)
            if not self.gui_state.is_2d_scene:
                changed, new_sensors = psim.SliderInt(
                    "Camera Positions",
                    self.gui_state.nb_sensors,
                    4, 128
                )
                if changed:
                    self.gui_state.nb_sensors = new_sensors
            
            # CFG Scale
            changed, new_cfg = psim.SliderFloat(
                "CFG Scale",
                self.gui_state.cfg_scale,
                1.0, 100.0,
                format="%.2f"
            )
            if changed:
                self.gui_state.cfg_scale = new_cfg
                # Update the model config
                self.sd.config['guidance_scale'] = new_cfg
            
            # Min Time
            changed, new_min_time = psim.SliderFloat(
                "Min Time",
                self.gui_state.min_time,
                0.0, 1.0,
                format="%.3f"
            )
            if changed:
                self.gui_state.min_time = new_min_time
                # Update the model config
                self.sd.config['min_time'] = new_min_time
            
            # Max Time
            changed, new_max_time = psim.SliderFloat(
                "Max Time",
                self.gui_state.max_time,
                0.0, 1.0,
                format="%.3f"
            )
            if changed:
                self.gui_state.max_time = new_max_time
                # Update the model config
                self.sd.config['max_time'] = new_max_time
            
            psim.Separator()
            
            # Save textures button
            if psim.Button("Save Textures"):
                self.save_textures()
            
            psim.PopItemWidth()
        
        return reset_rendering
    
    def handle_custom_input(self):
        """Handle optimization-specific keyboard shortcuts"""
        # Toggle optimization with 'O' key
        if psim.IsKeyPressed(psim.ImGuiKey_O):
            self.gui_state.is_optimizing = not self.gui_state.is_optimizing
            if self.gui_state.is_optimizing:
                print("Started optimization")
            else:
                print("Stopped optimization")
    
    def add_custom_hints(self, hints):
        """Add optimization-specific hints to the overlay"""
        hints.append(("O", "Toggle optimization"))
        hints.append(("Ctrl+S", "Save textures"))
    
    def save_textures(self):
        """Save optimized textures to disk"""
        output_dir = Path(f'outputs/{self.gui_state.scene_name}_tex')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Render current view
            dr_image = mi.render(self.scene, params=self.scene_params, seed=0)
            
            dr_image = dr_image / (dr_image + 1)
            dr_image = dr_image ** (1/2.2)
            
            mi.util.write_bitmap(
                str(output_dir / f'{self.gui_state.scene_name}_opt.exr'), dr_image
            )
            
            # Save individual texture parameters
            for k, v in self.opt.items():
                param_name = k.split(".")[0]
                mi.util.write_bitmap(
                    str(output_dir / f'{param_name}.exr'), v
                )
            
            print(f"Saved textures to {output_dir}")
        except Exception as e:
            print(f"Failed to save textures: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='GoTex - Interactive Texture Optimization with Stable Diffusion'
    )
    parser.add_argument(
        'scene',
        nargs='?',
        help='Scene module path (e.g., "scenes.dragon") or Python file path (e.g., "src/scenes/painting.py"). '
             'If omitted, uses default painting scene.'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Custom prompt for Stable Diffusion. If not provided, uses the prompt from the scene configuration.'
    )
    
    args = parser.parse_args()
    
    gui = GoTexGUI(args.scene, prompt_override=args.prompt)
    gui.run()


if __name__ == "__main__":
    main()
