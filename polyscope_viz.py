import polyscope as ps
import polyscope.imgui as psim
import numpy as np

import drjit as dr
import mitsuba as mi


def view(scene_params):
    ps.init()

    def get_property(meshes, param_key, name):
        if name in param_key:
            mesh_id = param_key.split('.')[0]
            if mesh_id not in meshes:
                meshes[mesh_id] = {name: scene_params[param_key]}
            else:
                meshes[mesh_id][name] = scene_params[param_key]

    meshes = {}
    for param_key in scene_params.keys():
        get_property(meshes, param_key, 'vertex_positions')
        get_property(meshes, param_key, 'faces')

    for mesh_id, mesh in meshes.items():
        faces = mesh['faces'].numpy()    
        postions = mesh['vertex_positions'].numpy()

        faces = np.reshape(faces, (len(faces) // 3, 3))
        postions = np.reshape(postions, (len(postions) // 3, 3))
        
        ps.register_surface_mesh(mesh_id, postions, faces)

    vertex_id = mi.UInt32(0, 1, 2, 3, 14, 15, 16, 40)
    #vertex_id = dr.arange(mi.UInt32, len(scene_params['Walls_0001.vertex_positions'])//3)
    vertices = dr.gather(mi.Point3f, scene_params['Walls_0001.vertex_positions'], vertex_id).numpy()
    vertices = np.transpose(vertices)

    ps.register_point_cloud("Frame", vertices)

    ps.show()
