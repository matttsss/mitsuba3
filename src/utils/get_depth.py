import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')
import sys
sys.path.insert(0, '/home/matteo/Documents/SemProj/GoTex/src')
from renderer import get_depth
from scenes.dragon import load_scene


scene, scene_params, _, _ = load_scene(512)
depth = get_depth(scene, scene.sensors()[0])

mi.util.write_bitmap("outputs/depth.exr", depth)