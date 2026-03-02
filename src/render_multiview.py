"""
Script to render the coffee maker scene from 100 random viewpoints
and save the outputs to the outputs directory.
"""
from __future__ import annotations

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
    scene, scene_params, scene_metadata = load_scene(render_size=1024)
    
    # Render from 100 random viewpoints
    nb_sensors = 100
    print(f"Rendering {nb_sensors} viewpoints...")
    for i in range(nb_sensors):
        # Randomize the sensor position
        randomize_sensor(
            sensor_idx=i,
            sensor_count=nb_sensors,
            scene_params=scene_params,
            sensor_to_world_key='sensor.to_world',
            target=scene_metadata['target'],
            radius=scene_metadata['radius']
        )
        
        # Render the scene
        image = mi.render(scene, params=scene_params)
        
        # Convert to numpy and save as EXR
        output_path = output_dir / f"view_{i:03d}.exr"
        
        mi.util.write_bitmap(str(output_path), image)
        
        if (i + 1) % 10 == 0:
            print(f"  Rendered {i + 1}/{nb_sensors} views")
    
    print(f"\nRendering complete! Images saved to {output_dir}")


if __name__ == "__main__":
    main()
