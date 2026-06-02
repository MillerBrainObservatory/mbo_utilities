"""Build an isoview microscope camera rig in Blender.

Run headless:
    blender -b -P isoview_rig.py

Produces isoview_4cam.blend and isoview_4cam.png next to this script.

Geometry: a flat optical table, a central vertical mounting cylinder with the
sample at its center, and N cameras spaced evenly around it, each on a pedestal
and pointed horizontally at the center. Adjacent cameras are 90 deg apart.
Mirror pairs (opposite cameras sharing one optical axis) are color-matched so
that merging them down to two orthogonal cameras reads visually.
"""

import math
import os

import bpy
from mathutils import Vector

CAM_DIST = 6.0       # camera distance from center axis
CAM_HEIGHT = 2.0     # camera / sample height above the table
CYL_RADIUS = 0.6
CYL_HEIGHT = 4.0
TABLE_SIZE = 22.0
LENS_LEN = 1.2
BODY_SIZE = 0.9

# color per mirror-pair axis: X-axis pair, Y-axis pair, then extras
PAIR_COLORS = [
    (0.85, 0.20, 0.20, 1.0),  # red   -> X axis pair
    (0.20, 0.45, 0.90, 1.0),  # blue  -> Y axis pair
    (0.20, 0.70, 0.30, 1.0),
    (0.85, 0.65, 0.15, 1.0),
]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                  bpy.data.lights, bpy.data.curves):
        for item in list(block):
            block.remove(item)


def material(name, color, emission=0.0, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def track_quat(forward, axis="-Z"):
    return Vector(forward).normalized().to_track_quat(axis, "Y")


def add_camera_unit(index, angle_deg, color):
    """Real camera + lens cone + body box + optical-axis rod, all aimed at center."""
    a = math.radians(angle_deg)
    pos = Vector((CAM_DIST * math.cos(a), CAM_DIST * math.sin(a), CAM_HEIGHT))
    target = Vector((0.0, 0.0, CAM_HEIGHT))
    fwd = (target - pos).normalized()

    cam_mat = material(f"cam{index}_mat", color, roughness=0.4)
    lens_mat = material(f"lens{index}_mat", (0.05, 0.05, 0.05, 1.0),
                        metallic=0.8, roughness=0.25)

    # pedestal (the table each camera sits on)
    bpy.ops.mesh.primitive_cube_add(location=(pos.x, pos.y, CAM_HEIGHT / 2.0))
    ped = bpy.context.active_object
    ped.scale = (0.8, 0.8, CAM_HEIGHT / 2.0)
    ped.name = f"pedestal{index}"
    assign(ped, material(f"ped{index}_mat", (0.3, 0.3, 0.33, 1.0), roughness=0.8))

    # camera body
    bpy.ops.mesh.primitive_cube_add(location=pos)
    body = bpy.context.active_object
    body.scale = (BODY_SIZE / 2.0, BODY_SIZE / 2.0, BODY_SIZE / 2.0)
    body.rotation_euler = track_quat(fwd, "-Z").to_euler()
    body.name = f"body{index}"
    assign(body, cam_mat)

    # objective / lens cone, apex pointing at the sample
    lens_pos = pos + fwd * (BODY_SIZE / 2.0 + LENS_LEN / 2.0)
    bpy.ops.mesh.primitive_cone_add(radius1=0.35, radius2=0.12, depth=LENS_LEN,
                                    location=lens_pos)
    lens = bpy.context.active_object
    lens.rotation_euler = track_quat(fwd, "Z").to_euler()  # +Z (apex) -> sample
    lens.name = f"lens{index}"
    assign(lens, lens_mat)

    # optical axis rod from lens to sample
    dist = (target - lens_pos).length
    mid = lens_pos + fwd * (dist / 2.0)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=dist, location=mid)
    rod = bpy.context.active_object
    rod.rotation_euler = track_quat(fwd, "Z").to_euler()
    rod.name = f"axis{index}"
    rod_mat = material(f"axis{index}_mat", color, emission=1.5)
    assign(rod, rod_mat)

    # actual Blender camera (shows the camera gizmo / direction in the viewport)
    cam_data = bpy.data.cameras.new(f"isoCam{index}")
    cam_data.lens = 35
    cam_obj = bpy.data.objects.new(f"isoCam{index}", cam_data)
    cam_obj.location = pos
    cam_obj.rotation_euler = track_quat(fwd, "-Z").to_euler()
    bpy.context.collection.objects.link(cam_obj)


def build_rig(n_cameras):
    clear_scene()

    # optical table
    bpy.ops.mesh.primitive_plane_add(size=TABLE_SIZE, location=(0, 0, 0))
    table = bpy.context.active_object
    table.name = "table"
    assign(table, material("table_mat", (0.12, 0.12, 0.14, 1.0), roughness=0.9))

    # central mounting cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=CYL_RADIUS, depth=CYL_HEIGHT,
                                        location=(0, 0, CYL_HEIGHT / 2.0))
    cyl = bpy.context.active_object
    cyl.name = "mount_cylinder"
    assign(cyl, material("mount_mat", (0.55, 0.55, 0.6, 1.0),
                         metallic=0.9, roughness=0.3))

    # sample at the convergence point
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.22, location=(0, 0, CAM_HEIGHT))
    sample = bpy.context.active_object
    sample.name = "sample"
    bpy.ops.object.shade_smooth()
    assign(sample, material("sample_mat", (1.0, 0.95, 0.6, 1.0), emission=6.0))

    for i in range(n_cameras):
        angle = i * (360.0 / n_cameras)
        color = PAIR_COLORS[(i % n_cameras) % 2] if n_cameras % 2 == 0 \
            else PAIR_COLORS[i % len(PAIR_COLORS)]
        # pair opposite cameras by axis: same color for i and i+n/2
        if n_cameras % 2 == 0:
            color = PAIR_COLORS[i % (n_cameras // 2)]
        add_camera_unit(i + 1, angle, color)


def add_lighting_and_view():
    # key sun
    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("sun", sun_data)
    sun.location = (8, -8, 16)
    sun.rotation_euler = track_quat(Vector((-0.5, 0.5, -1.0)), "-Z").to_euler()
    bpy.context.collection.objects.link(sun)

    # ambient world light
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.04, 0.04, 0.05, 1.0)
    bg.inputs["Strength"].default_value = 1.0

    # render / preview camera looking over the whole rig
    view_data = bpy.data.cameras.new("view_cam")
    view_data.lens = 40
    view = bpy.data.objects.new("view_cam", view_data)
    view_loc = Vector((CAM_DIST * 1.7, -CAM_DIST * 1.8, CAM_HEIGHT * 3.2))
    view.location = view_loc
    view.rotation_euler = track_quat(
        Vector((0, 0, CAM_HEIGHT)) - view_loc, "-Z").to_euler()
    bpy.context.collection.objects.link(view)
    bpy.context.scene.camera = view


def render(out_png):
    scene = bpy.context.scene
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            continue
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1000
    scene.render.film_transparent = False
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() \
        else os.getcwd()
    build_rig(4)
    add_lighting_and_view()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(here, "isoview_4cam.blend"))
    render(os.path.join(here, "isoview_4cam.png"))


if __name__ == "__main__":
    main()
