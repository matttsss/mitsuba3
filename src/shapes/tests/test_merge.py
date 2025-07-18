import mitsuba as mi

from mitsuba.scalar_rgb.test.util import fresolver_append_path


def example_mesh(**kwargs):
    return {
        "type" : "ply",
        "filename" : "resources/data/tests/ply/triangle.ply",
        "face_normals" : True,
        **kwargs
    }


@fresolver_append_path
def test01_merge_single_shape(variants_all_backends_once):
    # One shape on its own, no BSDF
    m = mi.load_dict({
        "type": "merge",
        "child1": example_mesh(),
    })
    assert isinstance(m, mi.Mesh)
    assert m.id() == "child1"

    # One shape in a scene, no BSDF
    m = mi.load_dict({
        "type": "scene",
        "parent": {
            "type": "merge",
            "child1": example_mesh(),
        },
    })
    assert len(m.shapes()) == 1
    assert m.shapes()[0].id() == "parent"

    # One shape in a scene, with a BSDF
    m = mi.load_dict({
        "type": "scene",
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "diffuse" }),
        },
    })
    assert len(m.shapes()) == 1
    assert m.shapes()[0].id() == "parent"

    # Non-mesh --> should be just passed through
    m = mi.load_dict({
        "type": "merge",
        "child1": {
            "type": "sphere",
        },
    })
    assert isinstance(m, mi.Shape)
    assert m.id() == "child1"


@fresolver_append_path
def test02_two_shapes(variants_all_rgb):
    # No BSDF --> doesn't merge
    m = mi.load_dict({
        "type": "merge",
        "child1": example_mesh(),
        "child2": example_mesh(),
    })
    assert len(m) == 2
    assert set([m[0].id(), m[1].id()]) == {"child1", "child2"}

    # No BSDF --> doesn't merge
    m = mi.load_dict({
        "type": "merge",
        "child1": example_mesh(bsdf={ "type": "diffuse" }),
        "child2": example_mesh(),
    })
    assert len(m) == 2
    assert set([m[0].id(), m[1].id()]) == {"child1", "child2"}

    # Same BSDF: merge
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
            "child2": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
        }
    })
    assert len(m.shapes()) == 1
    assert m.shapes()[0].id() == "parent"

    # Non-mesh --> doesn't merge
    m = mi.load_dict({
        "type": "scene",
        "bsdf1": { "type": "diffuse" },
        "parent": {
            "type": "merge",
            "child1": example_mesh(bsdf={ "type": "ref", "id": "bsdf1" }),
            "child2": {
                "type": "sphere",
                "bsdf": { "type": "ref", "id": "bsdf1" }
            },
        }
    })
    assert len(m.shapes()) == 2
    assert set([m.shapes()[0].id(), m.shapes()[1].id()]) == {"parent", "child2"}
