"""
Watch the flexible KUKA drag its tool across the worktable -- before vs after
ILC (the contact task).

    python src/render_contact.py            # outputs/kuka_contact.gif
    python src/render_contact.py --live     # interactive window

The green tool tip slides on the brown table along the yellow circle. With the
ILC off, friction + joint elasticity drag the tool off the circle; with the
learned correction on, it tracks the circle. Reuses the train/capture helpers
from render_kuka.py with the contact plant.
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glx")

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from contact_simulation import make_contact_plant, make_contact_reference  # noqa: E402
from render_kuka import train, capture_pass, SETTLE, HOLD  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
CAM = dict(lookat=[0.55, 0.0, 0.32], distance=1.0, azimuth=120.0, elevation=-32.0)


def make_cam():
    c = mujoco.MjvCamera()
    c.lookat[:] = CAM["lookat"]; c.distance = CAM["distance"]
    c.azimuth = CAM["azimuth"]; c.elevation = CAM["elevation"]
    return c


def render_gif(path=os.path.join(OUT, "kuka_contact.gif")):
    from PIL import Image
    ref = make_contact_reference(n=200)
    lambdas, xyz, qref = ref
    plant = make_contact_plant(marker_xyz=xyz[::3])
    print("training contact-task ILC ...")
    ilc = train(plant, ref)
    renderer = mujoco.Renderer(plant.model, height=540, width=720, max_geom=20000)
    cam = make_cam()
    frames = []
    print("rendering pass 1 (no ILC) ...")
    capture_pass(plant, ilc, ref, False, renderer, cam,
                 "NO ILC: friction drags the tool off the circle",
                 frames, trace_rgba=(1.0, 0.25, 0.2, 1.0))
    print("rendering pass 2 (after ILC) ...")
    capture_pass(plant, ilc, ref, True, renderer, cam,
                 "AFTER ILC: tool tracks the circle on the surface",
                 frames, trace_rgba=(0.2, 0.9, 0.3, 1.0))
    renderer.close()
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=50, loop=0)
    print(f"wrote {path}  ({len(imgs)} frames)")


def live():
    import mujoco.viewer
    ref = make_contact_reference(n=200)
    lambdas, xyz, qref = ref
    plant = make_contact_plant(marker_xyz=xyz[::3])
    print("training contact-task ILC ...")
    ilc = train(plant, ref)
    print("opening viewer: loops NO-ILC pass, then AFTER-ILC pass.")
    with mujoco.viewer.launch_passive(plant.model, plant.data) as viewer:
        viewer.cam.lookat[:] = CAM["lookat"]; viewer.cam.distance = CAM["distance"]
        viewer.cam.azimuth = CAM["azimuth"]; viewer.cam.elevation = CAM["elevation"]
        while viewer.is_running():
            for use_ilc in (False, True):
                plant.reset(qref[0])
                for _ in range(SETTLE):
                    u0 = ilc.correction_at(lambdas[0]) if use_ilc else np.zeros(7)
                    plant.command_motor(qref[0] + u0); viewer.sync()
                for idx in range(len(lambdas)):
                    if not viewer.is_running():
                        return
                    u = ilc.correction_at(lambdas[idx]) if use_ilc else np.zeros(7)
                    for _ in range(HOLD):
                        plant.command_motor(qref[idx] + u); viewer.sync()


if __name__ == "__main__":
    if "--live" in sys.argv:
        live()
    else:
        render_gif()
