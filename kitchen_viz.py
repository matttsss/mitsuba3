import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

def transform(t, s, center, vertices):
    # Scale with regards to the center of the window
    new_vertices = (vertices - center)
    new_vertices.x *= s.x
    new_vertices.y *= s.y
    new_vertices += center

    new_vertices.x += t.x
    new_vertices.y += t.y      

    return new_vertices

kitchen = mi.load_file("kitchen/scene_persp.xml", optimize=False, integrator='prb_projective')
kitchen_params = mi.traverse(kitchen)

mesh_key = 'Walls_0001.vertex_positions'
vertex_id = mi.UInt32(0, 1, 2, 3, 14, 15, 16, 40, 53)

t, s = mi.Vector2f(0.0), mi.Vector2f(1.)
dr.enable_grad(kitchen_params[mesh_key], t.x)

vertices = dr.gather(mi.Point3f, kitchen_params[mesh_key], vertex_id)
center = mi.Point3f(dr.mean(vertices, axis=1))

new_vertices = transform(t, s, center, vertices)

dr.scatter(kitchen_params[mesh_key], new_vertices, vertex_id)
kitchen_params.set_dirty(mesh_key)
kitchen_params.update()

dr.forward(t.x, dr.ADFlag.ClearEdges)


if True:
    image = mi.render(kitchen, kitchen_params, spp=512)
    grad = dr.forward_to(image)

    mi.util.write_bitmap("results/kitchen.exr", image)
    mi.util.write_bitmap("results/kitchen_grad.exr", grad)
else:
    import numpy as np
    import polyscope as ps
    from polyscope_viz import get_meshes

    ps.init()

    meshes = get_meshes(kitchen_params)

    for mesh_id, mesh in meshes.items():
        ps.register_surface_mesh(mesh_id, mesh['vertex_positions'], mesh['faces'])


    vertex_id = mi.UInt32(0, 1, 2, 3, 14, 15, 16, 40)
    vertices = dr.gather(mi.Point3f, kitchen_params['Walls_0001.vertex_positions'], vertex_id).numpy()
    vertices = np.transpose(vertices)

    ps.register_point_cloud("Frame", vertices)

    ps.show()
