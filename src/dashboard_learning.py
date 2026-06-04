"""
Real-time "mission control" dashboard of the KUKA path-ILC LEARNING process.

    python src/dashboard_learning.py        # writes outputs/kuka_learning_dashboard.mp4

The video runs trials 0..K in sequence. Four live panels:
  - left:        the KUKA tracing the path, tool-tip trace (deviation x8)
  - top-right:   TCP error vs path parameter -- current trial bold, past trials
                 faded, so you watch the error curve collapse trial by trial
  - mid-right:   the 7 ILC correction signals u(lambda) currently applied
  - bottom-right: convergence -- Cartesian RMS vs trial (log), a point per trial
A header shows the live trial number and RMS.
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
from kuka_plant import FlexArmPlant  # noqa: E402
from kuka_simulation import make_reference, new_ilc  # noqa: E402
from render_kuka import make_cam, _add_sphere, AMP, SETTLE, HOLD  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
N = 200
N_TRIALS = 5            # trials shown in the learning animation
RENDER_EVERY = 4        # render one frame every Nth path point (keeps it short)
J_COLORS = plt.cm.viridis(np.linspace(0, 0.92, 7))


def collect_trial(plant, ilc, ref, use_ilc, renderer, cam, trace_rgba):
    """Run one trial; capture rendered frames, lambda/error per frame, the
    applied u(lambda) table, and the trial RMS."""
    lambdas, xyz, qref = ref
    u_tab = np.array([ilc.correction_at(l) for l in lambdas])   # applied feedforward
    plant.reset(qref[0])
    for _ in range(SETTLE):
        u0 = ilc.correction_at(lambdas[0]) if use_ilc else np.zeros(7)
        plant.command_motor(qref[0] + u0)
    imgs, lam_s, err_s, trace = [], [], [], []
    all_err = []
    eq_h, lam_h = [], []
    for idx in range(len(lambdas)):
        u = ilc.correction_at(lambdas[idx]) if use_ilc else np.zeros(7)
        for k in range(HOLD):
            plant.command_motor(qref[idx] + u)
        actual = plant.read_tcp()
        e = np.linalg.norm(actual - xyz[idx]); all_err.append(e)
        eq_h.append(qref[idx] - plant.read_joint_encoders()); lam_h.append(lambdas[idx])
        if idx % RENDER_EVERY == 0:
            trace.append(xyz[idx] + AMP * (actual - xyz[idx]))
            renderer.update_scene(plant.data, camera=cam)
            scn = renderer.scene
            for p in trace:
                _add_sphere(scn, p, 0.012, trace_rgba)
            _add_sphere(scn, trace[-1], 0.02, (1, 1, 1, 1))
            imgs.append(renderer.render().copy())
            lam_s.append(lambdas[idx]); err_s.append(1e3 * e)
    rms = 1e6 * np.sqrt(np.mean(np.square(all_err)))
    return dict(imgs=imgs, lam=np.array(lam_s), err=np.array(err_s),
                u_tab=u_tab * 1e3, rms=rms,
                lam_h=np.array(lam_h), eq=np.array(eq_h))


def main(path=os.path.join(OUT, "kuka_learning_dashboard.mp4")):
    ref = make_reference("A", n=N)
    lambdas, xyz, qref = ref
    plant = FlexArmPlant(marker_xyz=xyz[::3])
    renderer = mujoco.Renderer(plant.model, height=620, width=620, max_geom=20000)
    cam = make_cam()

    ilc = new_ilc(N)
    trials = []
    for t in range(N_TRIALS):
        print(f"collecting trial {t} ...")
        tr = collect_trial(plant, ilc, ref, use_ilc=(t > 0), renderer=renderer,
                           cam=cam, trace_rgba=(0.2, 0.9, 0.3, 1.0))
        trials.append(tr)
        ilc.update_from_trial(tr["lam_h"], tr["eq"])
    renderer.close()

    emax = max(tr["err"].max() for tr in trials) * 1.1
    umax = max(np.abs(tr["u_tab"]).max() for tr in trials) * 1.1 + 1
    rms_all = [tr["rms"] for tr in trials]
    frames = [(t, k) for t in range(N_TRIALS) for k in range(len(trials[t]["imgs"]))]

    fig = plt.figure(figsize=(13.5, 7.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], hspace=0.55, wspace=0.22)
    ax_im = fig.add_subplot(gs[:, 0]); ax_im.axis("off")
    ax_e = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, 1])
    im_artist = ax_im.imshow(trials[0]["imgs"][0])
    header = fig.suptitle("", fontsize=14, fontweight="bold")

    def update(f):
        t, k = frames[f]
        tr = trials[t]
        im_artist.set_data(tr["imgs"][k])

        ax_e.cla()
        for s in range(t):                       # faded past trials
            ax_e.plot(trials[s]["lam"], trials[s]["err"], color="0.7", lw=1)
        ax_e.plot(tr["lam"][:k + 1], tr["err"][:k + 1], color="#1a9850", lw=2.2)
        ax_e.set_xlim(0, 1); ax_e.set_ylim(0, emax)
        ax_e.set_ylabel("TCP error [mm]"); ax_e.set_xlabel("path parameter λ")
        ax_e.set_title("Tracking error along path (grey = past trials)")
        ax_e.grid(alpha=0.3)

        ax_u.cla()
        for j in range(7):
            ax_u.plot(lambdas, tr["u_tab"][:, j], color=J_COLORS[j], lw=1.3)
        ax_u.axvline(tr["lam"][k], color="0.5", ls=":", lw=1)
        ax_u.set_xlim(0, 1); ax_u.set_ylim(-umax, umax)
        ax_u.set_ylabel("ILC input [mrad]"); ax_u.set_xlabel("path parameter λ")
        ax_u.set_title("Learned feedforward applied this trial ($u_1..u_7$)")
        ax_u.grid(alpha=0.3)

        ax_c.cla()
        ax_c.semilogy(range(t + 1), rms_all[:t + 1], "o-", color="#0072B2")
        ax_c.semilogy([t], [rms_all[t]], "o", color="#D55E00", ms=10)
        ax_c.set_xlim(-0.3, N_TRIALS - 0.7)
        ax_c.set_xticks(range(N_TRIALS))
        ax_c.set_ylabel("RMS [µm] (log)"); ax_c.set_xlabel("trial")
        ax_c.set_title("Convergence"); ax_c.grid(alpha=0.3, which="both")

        header.set_text(f"KUKA iiwa14 path-ILC learning   —   "
                        f"trial {t}/{N_TRIALS-1}    RMS = {tr['rms']:.0f} µm"
                        f"{'   (no correction yet)' if t == 0 else ''}")
        return im_artist,

    print(f"writing {len(frames)} frames -> {path} ...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(path, writer=animation.FFMpegWriter(fps=12, bitrate=2800))
    plt.close(fig)
    print(f"wrote {path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
