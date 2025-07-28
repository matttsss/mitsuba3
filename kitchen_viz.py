import igl
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

kitchen = mi.load_file("kitchen/scene.xml", optimize=False)
kitchen_params = mi.traverse(kitchen)

mesh_key = 'Walls_0001.vertex_positions'
vertex_id = mi.UInt32(0, 1, 2, 3, 14, 15, 16, 40, 53)

vertices = dr.gather(mi.Point3f, kitchen_params[mesh_key], vertex_id)
center = mi.Point3f(dr.mean(vertices, axis=1))

t, s = mi.Vector2f(0.1), mi.Vector2f(1.25)
new_vertices = transform(t, s, center, vertices)

dr.scatter(kitchen_params[mesh_key], new_vertices, vertex_id)
kitchen_params.set_dirty(mesh_key)
kitchen_params.update()

if True:
    image = mi.render(kitchen, kitchen_params, spp=128)
    mi.util.write_bitmap("results/kitchen.exr", image)
else:
    from polyscope_viz import view
    view(kitchen_params)
