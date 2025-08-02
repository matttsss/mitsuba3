import numpy as np

import drjit as dr
import mitsuba as mi

rect_faces = np.array([
    [1, 2, 0],
    [1, 3, 2]
])
rect_pos = np.array([
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 1, 0]
])

swap_axis = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])

def update_rectangle(mi_params, ps_rectangle, height, scale):
    transform = swap_axis @ np.array([
        [scale[1], 0, 0],
        [0,  1, 0],
        [0, 0, scale[0]]
    ])

    offset = np.array([0, height, 0])
    ps_rectangle.update_vertex_positions(
        rect_pos @ transform + offset
    )

    mi_transform = np.concatenate((transform, offset.reshape((3, 1))), axis=1)
    mi_transform = np.concatenate((mi_transform, np.array([[0, 0, 0, 1]])), axis=0, dtype=np.float32)

    mi_params['SensorPlane.to_world'] = mi.AffineTransform4f(mi_transform)


def get_rectangle(scene_params, mesh_key):
    to_world = scene_params[mesh_key + '.to_world']
    offset = np.squeeze(to_world.translation())
    to_world = to_world.matrix.numpy()
    to_world = np.squeeze(to_world)[:3, :3]

    positions = rect_pos @ to_world

    return {
        'vertex_positions': positions + offset,
        'faces': rect_faces
    }


def get_meshes(scene_params):
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

    for mesh_id in meshes:
        faces = meshes[mesh_id]['faces'].numpy()    
        positions = meshes[mesh_id]['vertex_positions'].numpy()

        faces = np.reshape(faces, (len(faces) // 3, 3))
        positions = np.reshape(positions, (len(positions) // 3, 3))

        meshes[mesh_id]['faces'] = faces    
        meshes[mesh_id]['vertex_positions'] = positions
    
    return meshes
