"""
Watch the flexible KUKA iiwa14 trace a 3D path -- before vs after ILC.

    python src/render_kuka.py            # render outputs/kuka_tracking.gif (no GUI)
    python src/render_kuka.py --live     # open an interactive MuJoCo window

The yellow dots are the desired 3D path. With the ILC correction OFF the arm's
tool drifts off the path (joint elasticity / drivetrain deflection it doesn't
know about); with the learned correction ON the tool tracks the path. The arm,
meshes, and dynamics are the real Menagerie KUKA; only the drivetrain
compliance is added (see kuka_plant.py).
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glx")

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from kuka_plant import FlexArmPlant  # noqa: E402
from kuka_simulation import make_reference, new_ilc  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

N = 200
HOLD = 30
SETTLE = 500
N_TRIALS = 7
CAM = dict(lookat=[0.45, 0.0, 0.45], distance=1.7, azimuth=140.0, elevation=-18.0)


def make_cam():
    c = mujoco.MjvCamera()
    c.lookat[:] = CAM["lookat"]; c.distance = CAM["distance"]
    c.azimuth = CAM["azimuth"]; c.elevation = CAM["elevation"]
    return c


def train(plant, ref):
    lambdas, xyz, qref = ref
    ilc = new_ilc(len(lambdas))
    for i in range(N_TRIALS):
        plant.reset(qref[0])
        for _ in range(SETTLE):
            u0 = ilc.correction_at(lambdas[0]) if i > 0 else np.zeros(7)
            plant.command_motor(qref[0] + u0)
        lam_h, eq_h = [], []
        for idx in range(len(lambdas)):
            u = ilc.correction_at(lambdas[idx]) if i > 0 else np.zeros(7)
            for _ in range(HOLD):
                plant.command_motor(qref[idx] + u)
            lam_h.append(lambdas[idx])
            eq_h.append(qref[idx] - plant.read_joint_encoders())
        ilc.update_from_trial(np.array(lam_h), np.array(eq_h))
    return ilc


AMP = 8.0   # path-deviation amplification for the trace (cf. paper Fig. 4: 50x)


def _add_sphere(scn, pos, size, rgba):
    """Append one sphere geom to a render scene (decoration, no physics)."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_SPHERE),
                        np.array([size, 0, 0], dtype=np.float64),
                        np.asarray(pos, dtype=np.float64),
                        np.eye(3).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def capture_pass(plant, ilc, ref, use_ilc, renderer, cam, label, frames,
                 hold_render=6, trace_rgba=(1.0, 0.25, 0.2, 1.0)):
    """Render a pass, drawing the tool-tip trace with its deviation from the
    desired path amplified by AMP, so the (otherwise sub-cm) error is visible:
    no-ILC balloons off the yellow path, after-ILC hugs it."""
    from PIL import Image, ImageDraw
    lambdas, xyz, qref = ref
    plant.reset(qref[0])
    for _ in range(SETTLE):
        u0 = ilc.correction_at(lambdas[0]) if use_ilc else np.zeros(7)
        plant.command_motor(qref[0] + u0)
    trace = []
    for idx in range(len(lambdas)):
        u = ilc.correction_at(lambdas[idx]) if use_ilc else np.zeros(7)
        for k in range(HOLD):
            plant.command_motor(qref[idx] + u)
            if k == hold_render and idx % 2 == 0:   # one frame per 2 path pts
                # amplified-deviation trace point: desired + AMP*(actual-desired)
                actual = plant.read_tcp()
                trace.append(xyz[idx] + AMP * (actual - xyz[idx]))
                renderer.update_scene(plant.data, camera=cam)
                scn = renderer.scene
                for p in trace:
                    _add_sphere(scn, p, 0.012, trace_rgba)
                _add_sphere(scn, trace[-1], 0.02, (1, 1, 1, 1))  # current tip
                img = renderer.render().copy()
                pim = Image.fromarray(img); d = ImageDraw.Draw(pim)
                full = f"{label}   (deviation x{AMP:.0f})"
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    d.text((10 + dx, 8 + dy), full, fill=(0, 0, 0))
                d.text((10, 8), full, fill=(255, 245, 120))
                frames.append(np.asarray(pim))


def render_gif(path=os.path.join(OUT, "kuka_tracking.gif")):
    from PIL import Image
    ref = make_reference("A", n=N)
    lambdas, xyz, qref = ref
    plant = FlexArmPlant(marker_xyz=xyz[::3])
    print("training ILC on the base path ...")
    ilc = train(plant, ref)
    renderer = mujoco.Renderer(plant.model, height=540, width=720, max_geom=20000)
    cam = make_cam()
    frames = []
    print("rendering pass 1 (no ILC) ...")
    capture_pass(plant, ilc, ref, False, renderer, cam,
                 "NO ILC: tool drifts off the path (drivetrain deflection)",
                 frames, trace_rgba=(1.0, 0.25, 0.2, 1.0))
    print("rendering pass 2 (after ILC) ...")
    capture_pass(plant, ilc, ref, True, renderer, cam,
                 "AFTER ILC: tool tracks the path (learned, no tracker)",
                 frames, trace_rgba=(0.2, 0.9, 0.3, 1.0))
    renderer.close()
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=50, loop=0)
    print(f"wrote {path}  ({len(imgs)} frames)")


def live():
    import mujoco.viewer
    ref = make_reference("A", n=N)
    lambdas, xyz, qref = ref
    plant = FlexArmPlant(marker_xyz=xyz[::3])
    print("training ILC on the base path ...")
    ilc = train(plant, ref)
    print("opening viewer: loops NO-ILC pass, then AFTER-ILC pass.")
    with mujoco.viewer.launch_passive(plant.model, plant.data) as viewer:
        for k, v in CAM.items():
            setattr(viewer.cam, k, v) if k != "lookat" else \
                viewer.cam.lookat.__setitem__(slice(None), v)
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
