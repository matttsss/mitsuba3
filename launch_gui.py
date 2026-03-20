import argparse

from gotex.gui.gotex_gui import GoTexGUI

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
