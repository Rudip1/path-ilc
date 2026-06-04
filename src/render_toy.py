"""
Watch the arm trace the path -- before vs after ILC.

Two modes:

    python src/render_toy.py            # render outputs/toy_arm_tracking.gif (no GUI needed)
    python src/render_toy.py --live     # open an interactive MuJoCo window

What you see: the black dotted curve is the DESIRED Cartesian path. The arm
tip (green site) tries to follow it. In the first pass the ILC correction is
OFF (trial 0) and the tip visibly drifts off the curve because of the
unmodeled friction/gravity/compliance. In the second pass the learned ILC
correction is ON and the tip hugs the curve.

Rendering backend: we force GLX, which is the one that works on this machine
(EGL/OSMesa fail here). The interactive --live mode uses mujoco.viewer.
"""

import os
import sys

# Pick a working GL backend BEFORE importing mujoco's renderer.
os.environ.setdefault("MUJOCO_GL", "glx")

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from toy_arm import _ARM_XML, inverse_kinematics  # noqa: E402
from toy_simulation import make_reference, new_ilc        # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

SPEED = 0.4
SETTLE = 150
N_TRIALS = 7


def build_model_with_path(xz_ref, every=8):
    """Return an MjModel of the arm with the desired path drawn as small
    static spheres (visual only) so you can see what the tip should follow."""
    markers = []
    for i in range(0, len(xz_ref), every):
        x, z = xz_ref[i]
        markers.append(
            f'<geom type="sphere" size="0.008" pos="{x:.4f} 0 {z:.4f}" '
            f'rgba="1.0 0.85 0.1 0.9" contype="0" conaffinity="0"/>'
        )
    inject = "\n    ".join(markers)
    xml = _ARM_XML.replace("</worldbody>", f"    {inject}\n  </worldbody>")
    # Brighten the scene and give the arm a clearer color (visual only -- no
    # physics change) so the rendered GIF reads well.
    visual = ('<visual>'
              '<headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7" '
              'specular="0.1 0.1 0.1"/>'
              '<rgba haze="0.9 0.95 1 1"/>'
              '</visual>')
    xml = xml.replace("<option", visual + "\n  <option")
    xml = xml.replace('density="800"', 'density="800" rgba="0.30 0.55 0.85 1"')
    return mujoco.MjModel.from_xml_string(xml)


def look_at_plane(model):
    """A camera looking at the x-z plane (along -y) framed on the workspace."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, 0.0, 0.15]
    cam.distance = 1.2
    cam.azimuth = 90.0       # look along the -y axis at the x-z plane
    cam.elevation = -8.0
    return cam


def train(model):
    """Quick ILC training on path A using THIS model (with path markers)."""
    lambdas, xz, qref = make_reference("A", n=400)
    ilc = new_ilc(len(lambdas))
    data = mujoco.MjData(model)

    def trial(use_ilc):
        data.qpos[:3] = qref[0]; data.qvel[:3] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(SETTLE):
            u0 = ilc.correction_at(lambdas[0]) if use_ilc else np.zeros(3)
            data.ctrl[:3] = qref[0] + u0
            mujoco.mj_step(model, data)
        n = len(lambdas)
        visit = np.linspace(0, n - 1, max(int(round(n / SPEED)), 10)).round().astype(int)
        lam_h, eq_h = [], []
        for idx in visit:
            u = ilc.correction_at(lambdas[idx]) if use_ilc else np.zeros(3)
            data.ctrl[:3] = qref[idx] + u
            mujoco.mj_step(model, data)
            lam_h.append(lambdas[idx])
            eq_h.append(qref[idx] - np.array(data.sensordata[:3]))
        return np.array(lam_h), np.array(eq_h)

    for i in range(N_TRIALS):
        lam_h, eq_h = trial(use_ilc=(i > 0))
        ilc.update_from_trial(lam_h, eq_h)
    return ilc, (lambdas, xz, qref)


def run_capture(model, ilc, ref, use_ilc, renderer, cam, label, frames,
                stride=4):
    """Drive one trial and capture frames into `frames` (list of RGB arrays)."""
    from PIL import Image, ImageDraw
    lambdas, xz, qref = ref
    data = mujoco.MjData(model)
    data.qpos[:3] = qref[0]; data.qvel[:3] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(SETTLE):
        u0 = ilc.correction_at(lambdas[0]) if use_ilc else np.zeros(3)
        data.ctrl[:3] = qref[0] + u0
        mujoco.mj_step(model, data)
    n = len(lambdas)
    visit = np.linspace(0, n - 1, max(int(round(n / SPEED)), 10)).round().astype(int)
    for k, idx in enumerate(visit):
        u = ilc.correction_at(lambdas[idx]) if use_ilc else np.zeros(3)
        data.ctrl[:3] = qref[idx] + u
        mujoco.mj_step(model, data)
        if k % stride == 0:
            renderer.update_scene(data, camera=cam)
            img = renderer.render().copy()
            pim = Image.fromarray(img)
            d = ImageDraw.Draw(pim)
            # dark outline + bright text so the label reads on any background
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                d.text((10 + dx, 8 + dy), label, fill=(0, 0, 0))
            d.text((10, 8), label, fill=(255, 245, 120))
            frames.append(np.asarray(pim))


def render_gif(path=os.path.join(OUT, "toy_arm_tracking.gif")):
    from PIL import Image
    # train on a plain model, then visualize on the marker model
    lambdas, xz, qref = make_reference("A", n=400)
    model = build_model_with_path(xz)
    ilc, ref = train(model)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = look_at_plane(model)
    frames = []
    run_capture(model, ilc, ref, False, renderer, cam,
                "Pass 1: NO ILC (trial 0) - tip drifts off the path", frames)
    run_capture(model, ilc, ref, True, renderer, cam,
                "Pass 2: AFTER ILC - tip tracks the path", frames)
    renderer.close()

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=40, loop=0)
    print(f"wrote {path}  ({len(imgs)} frames)")


def live():
    import mujoco.viewer
    lambdas, xz, qref = make_reference("A", n=400)
    model = build_model_with_path(xz)
    ilc, ref = train(model)
    lambdas, xz, qref = ref
    data = mujoco.MjData(model)

    n = len(lambdas)
    visit = np.linspace(0, n - 1, max(int(round(n / SPEED)), 10)).round().astype(int)
    print("Opening viewer. The window loops: NO-ILC pass, then AFTER-ILC pass.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.55, 0.0, 0.15]
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -8.0
        while viewer.is_running():
            for use_ilc in (False, True):
                data.qpos[:3] = qref[0]; data.qvel[:3] = 0.0
                mujoco.mj_forward(model, data)
                for _ in range(SETTLE):
                    u0 = ilc.correction_at(lambdas[0]) if use_ilc else np.zeros(3)
                    data.ctrl[:3] = qref[0] + u0
                    mujoco.mj_step(model, data)
                    viewer.sync()
                for idx in visit:
                    if not viewer.is_running():
                        return
                    u = ilc.correction_at(lambdas[idx]) if use_ilc else np.zeros(3)
                    data.ctrl[:3] = qref[idx] + u
                    mujoco.mj_step(model, data)
                    viewer.sync()


if __name__ == "__main__":
    if "--live" in sys.argv:
        live()
    else:
        render_gif()
