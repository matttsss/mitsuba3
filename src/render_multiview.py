"""
Script to render the coffee maker scene from 100 random viewpoints
and save the outputs to the outputs directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import drjit as dr
import mitsuba as mi

from scenes.coffee_maker import load_scene
from renderer import randomize_sensor


def main():
    # Create output directory
    output_dir = Path("outputs/coffee_maker_multiview")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mi.set_variant("cuda_ad_rgb")
    
    # Load the scene
    print("Loading coffee maker scene...")
    scene, scene_params = load_scene(render_size=1024)
    
    # Create a random number generator
    rng = dr.rng(seed=42)
    
    # Render from 100 random viewpoints
    print("Rendering 100 viewpoints...")
    for i in range(100):
        # Randomize the sensor position
        randomize_sensor(
            generator=rng,
            scene_params=scene_params,
            sensor_to_world_key='sensor.to_world',
            target=[0, 0.2, 0],
            radius=1.0
        )
        
        # Render the scene
        image = mi.render(scene, params=scene_params)
        
        # Convert to numpy and save as EXR
        output_path = output_dir / f"view_{i:03d}.exr"
        
        mi.util.write_bitmap(str(output_path), image)
        
        if (i + 1) % 10 == 0:
            print(f"  Rendered {i + 1}/100 views")
    
    print(f"\nRendering complete! Images saved to {output_dir}")


if __name__ == "__main__":
    main()
