import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')
ps.init()

from polyscope_viz import get_meshes, get_rectangle, update_rectangle


kitchen = mi.load_file('kitchen/scene.xml')
kitchen_params = mi.traverse(kitchen)

meshes = get_meshes(kitchen_params)

for mesh_id, mesh in meshes.items():
    ps.register_surface_mesh(mesh_id, mesh['vertex_positions'], mesh['faces'])

sensor = get_rectangle(kitchen_params, 'SensorPlane')
sensor = ps.register_surface_mesh('SensorPlane', sensor['vertex_positions'], sensor['faces'])
sensor.set_transparency(0.5)

vertex_id = mi.UInt32(0, 1, 2, 3, 14, 15, 16, 40)
vertices = dr.gather(mi.Point3f, kitchen_params['Walls_0001.vertex_positions'], vertex_id).numpy()
vertices = np.transpose(vertices)

ps.register_point_cloud("Frame", vertices)


sensor_height = 1.0
scale = [2.448, 2.445]
update_rectangle(kitchen_params, sensor, sensor_height, scale)
kitchen_params.update()

def callback():
    global sensor_height, scale

    updated_sensor_height, sensor_height = psim.SliderFloat("Sensor height", sensor_height, 0.0, 10.0)
    updated_sensor_scale, scale = psim.SliderFloat2("Sensor scale", scale, 0.1, 5)

    if updated_sensor_height or updated_sensor_scale:
        update_rectangle(kitchen_params, sensor, sensor_height, scale)
        kitchen_params.update()

    if (psim.Button("Render")):
        image = mi.render(kitchen, spp=1024)
        mi.util.write_bitmap('results/kitchen_irrad.exr', image)

ps.set_user_callback(callback)
ps.show()