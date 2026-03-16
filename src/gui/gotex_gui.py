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

from trainer import Trainer
from gui_base import MitsubaGUI, BaseGUIState


@dataclass
class GoTexGUIState(BaseGUIState):
    """Extended GUI state for optimization"""
    # Optimization controls
    is_optimizing: bool = False
    nb_sensors: int = 6

    # Scene configuration
    scene_name: str = "painting"
    is_2d_scene: bool = True


class GoTexGUI(MitsubaGUI):
    """
    Extended Mitsuba GUI with integrated texture optimization using Stable Diffusion.
    """
    
    def __init__(
        self,
        scene_path=None,
        prompt_override=None,
        texture_dir=None,
        program_name="GoTex - Texture Optimization"
    ):
        # Initialize custom GUI state before calling parent constructor
        self.gui_state = GoTexGUIState()
        
        self.prompt_override = prompt_override
        self.texture_dir_override = texture_dir
        
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
            
        scene_config = load_scene(512, texture_dir=self.texture_dir_override)
        scene = mi.load_dict(scene_config['scene'], optimize=False)

        self.scene_config = scene_config
        self.gui_state.scene_name = scene_config['scene_name']
        self.gui_state.is_2d_scene = scene_config['camera_config']['is_2d']

        # Create integrator (scene loader returns it in the dict, but we need separate)
        integrator = self.create_integrator("path")
        
        return scene, integrator
    
    def custom_init_setup(self):
        """Initialize optimization components"""
        # Set random seeds
        seed = 40
        random.seed(seed)
        torch.manual_seed(seed)

        sd_config = self.scene_config['sd_config']
        
        # Override prompt if provided
        if self.prompt_override is not None:
            sd_config['prompt'] = self.prompt_override
            print(f"Using custom prompt: {self.prompt_override}")
        
        self.trainer = Trainer(
            scene_params=self.scene_params,
            camera_config=self.scene_config['camera_config'],
            sd_config=sd_config,
            device=self.torch_device,
            seed=seed
        )

        self.trainer.setup_opt_sensors(6, 512)

        del self.scene_config
    
    def custom_gui_callback_step(self):
        """Perform optimization step if enabled"""
        if not self.gui_state.is_optimizing:
            return
        
        try:
            self.trainer.step(self.scene, self.scene_params)
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
                if psim.Button("Pause Optimization (O)"):
                    self.gui_state.is_optimizing = False
                psim.PopStyleColor(3)
            else:
                psim.PushStyleColor(psim.ImGuiCol_Button, (0.20, 0.32, 0.25, 0.6))
                psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, (0.25, 0.40, 0.31, 0.9))
                psim.PushStyleColor(psim.ImGuiCol_ButtonActive, (0.18, 0.29, 0.23, 1.0))
                if psim.Button("Start Optimization (O)"):
                    self.gui_state.is_optimizing = True
                psim.PopStyleColor(3)
            
            psim.SameLine()
            if psim.Button("Reset Optimizer"):
                raise NotImplementedError("Resetting optimizer is not implemented yet")
                reset_rendering = True
            
            # Optimization progress
            psim.Text(f"Step: {self.trainer.step_idx}")
            psim.SameLine()
            psim.Text(f"| Loss: {self.trainer.ema_loss:.6f}")
            
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
            for k, v in self.trainer.opt.items():
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
    parser.add_argument(
        '--texture-dir',
        type=str,
        default=None,
        help='Directory containing per-object textures (e.g., dragon.exr, base.exr, sword.exr). '
             'Passed to scene loaders that support texture_dir.'
    )
    
    args = parser.parse_args()
    
    gui = GoTexGUI(args.scene, prompt_override=args.prompt, texture_dir=args.texture_dir)
    gui.run()


if __name__ == "__main__":
    main()
