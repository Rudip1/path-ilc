"""
Thermal-adaptation dashboard: why a self-learning (online) layer is needed.

    python src/dashboard_thermal.py     # writes outputs/kuka_thermal_dashboard.mp4

As the robot warms up over many trials (non-repetitive thermal drift), a FROZEN
ILC table goes stale and the error creeps back up, while an ONLINE (continuously
updating) ILC tracks the drift. Panels: the arm tracing (frozen run, tool trace
deviation x8), the RMS-vs-trial for frozen vs online, and the rising joint
temperature.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

os.environ.setdefault("MUJOCO_GL", "glx")
sys.path.insert(0, os.path.dirname(__file__))
import mujoco  # noqa: E402
from kuka_plant import FlexArmPlant, Effects  # noqa: E402
from kuka_simulation import make_reference, run_trial, new_ilc  # noqa: E402
from render_kuka import make_cam, _add_sphere, AMP, SETTLE, HOLD  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
N, T, FREEZE = 200, 12, 5
RENDER_EVERY = 6


def frozen_trial_render(plant, ilc, ref, renderer, cam):
    """Run one trial on the frozen plant, capturing arm frames + trace + rms."""
    lambdas, xyz, qref = ref
    plant.reset(qref[0])
    for _ in range(SETTLE):
        plant.command_motor(qref[0] + ilc.correction_at(lambdas[0]))
    imgs, trace, errs, eq_h, lam_h = [], [], [], [], []
    for idx in range(len(lambdas)):
        u = ilc.correction_at(lambdas[idx])
        for _ in range(HOLD):
            plant.command_motor(qref[idx] + u)
        actual = plant.read_tcp(); errs.append(np.linalg.norm(actual - xyz[idx]))
        eq_h.append(qref[idx] - plant.read_joint_encoders()); lam_h.append(lambdas[idx])
        if idx % RENDER_EVERY == 0:
            trace.append(xyz[idx] + AMP * (actual - xyz[idx]))
            renderer.update_scene(plant.data, camera=cam)
            scn = renderer.scene
            for p in trace:
                _add_sphere(scn, p, 0.012, (0.9, 0.3, 0.2, 1))
            _add_sphere(scn, trace[-1], 0.02, (1, 1, 1, 1))
            imgs.append(renderer.render().copy())
    rms = 1e6 * np.sqrt(np.mean(np.square(errs)))
    return imgs, rms, plant.read_temperature().max(), np.array(lam_h), np.array(eq_h)


def main(path=os.path.join(OUT, "kuka_thermal_dashboard.mp4")):
    ref = make_reference("A", n=N)
    lambdas, xyz, qref = ref
    fx = lambda: Effects(thermal=True)

    # frozen plant (rendered) and online plant (data only), stepped trial-by-trial
    p_fr = FlexArmPlant(effects=fx(), marker_xyz=xyz[::3])
    p_on = FlexArmPlant(effects=fx())
    ilc_fr, ilc_on = new_ilc(N), new_ilc(N)
    renderer = mujoco.Renderer(p_fr.model, height=560, width=560, max_geom=20000)
    cam = make_cam()

    trial_imgs, rms_fr, rms_on, temp = [], [], [], []
    for t in range(T):
        print(f"trial {t} ...")
        imgs, r_fr, tmax, lam_h, eq_h = frozen_trial_render(
            p_fr, ilc_fr, ref, renderer, cam)
        if t < FREEZE:                       # frozen table stops learning at FREEZE
            ilc_fr.update_from_trial(lam_h, eq_h)
        r = run_trial(p_on, ilc_on, lambdas, xyz, qref, hold=HOLD,
                      settle=SETTLE, use_ilc=(t > 0))
        ilc_on.update_from_trial(r["lambdas"], r["eq"])
        trial_imgs.append(imgs); rms_fr.append(r_fr); rms_on.append(r["rms_um"])
        temp.append(tmax)
    renderer.close()

    frames = [(t, k) for t in range(T) for k in range(len(trial_imgs[t]))]
    fig = plt.figure(figsize=(12.8, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.4, wspace=0.22)
    ax_im = fig.add_subplot(gs[:, 0]); ax_im.axis("off")
    ax_r = fig.add_subplot(gs[0, 1]); ax_t = fig.add_subplot(gs[1, 1])
    im_art = ax_im.imshow(trial_imgs[0][0])
    header = fig.suptitle("", fontsize=14, fontweight="bold")

    def update(f):
        t, k = frames[f]
        im_art.set_data(trial_imgs[t][k])
        ax_r.cla()
        ax_r.semilogy(range(t + 1), rms_fr[:t + 1], "o-", color="#cc2222",
                      label="frozen ILC")
        ax_r.semilogy(range(t + 1), rms_on[:t + 1], "s-", color="#1a9850",
                      label="online ILC")
        ax_r.axvline(FREEZE - 1, color="0.6", ls=":", lw=1)
        ax_r.set_xlim(-0.3, T - 0.7); ax_r.set_ylabel("RMS [µm] (log)")
        ax_r.set_xlabel("trial"); ax_r.set_title("Accuracy vs trial")
        ax_r.legend(loc="upper left", fontsize=8); ax_r.grid(alpha=0.3, which="both")
        ax_t.cla()
        ax_t.plot(range(t + 1), temp[:t + 1], "o-", color="#b8860b")
        ax_t.set_xlim(-0.3, T - 0.7); ax_t.set_ylim(0, max(temp) * 1.1)
        ax_t.set_xlabel("trial"); ax_t.set_ylabel("joint temp (a.u.)")
        ax_t.set_title("Thermal warm-up"); ax_t.grid(alpha=0.3)
        phase = "learning" if t < FREEZE else "FROZEN table — drifting"
        header.set_text(f"Thermal drift & adaptation   —   trial {t}/{T-1}   "
                        f"({phase})   frozen={rms_fr[t]:.0f}µm  online={rms_on[t]:.0f}µm")
        return im_art,

    print(f"writing {len(frames)} frames -> {path} ...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(path, writer=animation.FFMpegWriter(fps=12, bitrate=2800))
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
