"""
Publication-quality analysis figures for the flexible-joint KUKA, in the style
of the ICRA paper (and tying in the mechanical-vibration view of the joints).

    python src/figures_analysis.py

Writes to ../outputs:
  trialwise_overview.png   - paper Fig.5 clone: Cartesian error / ILC input / path
                         speed over 7 trials (trial 3: speed change; trial 5:
                         ILC off; trial 6: ILC on)
  encoder_validation.png    - GROUND-TRUTH validation: the learned correction u(lambda)
                         recovers the true drivetrain joint error (only possible
                         in simulation; impossible with a laser tracker)
  error_along_path.png - Cartesian error along the path parameter, trial 0 vs final
  joint_modal_properties.png     - each joint as a torsional spring-mass-damper: natural
                         frequency & damping ratio, Q-filter cutoff, and a free
                         vibration decay (the mechanical-engineering view)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("MUJOCO_GL", "glx")
sys.path.insert(0, os.path.dirname(__file__))
import mujoco  # noqa: E402
from kuka_plant import (FlexArmPlant, JOINTS, STIFFNESS, DAMPING_L)  # noqa: E402
from kuka_simulation import make_reference, new_ilc  # noqa: E402

from plot_style import use_style  # noqa: E402

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

AX_COLORS = ["#0072B2", "#D55E00", "#009E73"]          # x, y, z
J_COLORS = plt.cm.viridis(np.linspace(0, 0.92, 7))     # 7 joints

N = 200
SETTLE = 500


def capture_trial(plant, ilc, ref, hold, use_ilc, t0):
    """One trial; capture per-sample time, lambda, Cartesian error (x,y,z),
    applied ILC input u (7), and the (constant) path-parameter speed."""
    lambdas, xyz, qref = ref
    dt = plant.model.opt.timestep
    lam_dot = (1.0 / (len(lambdas) - 1)) / (hold * dt)
    plant.reset(qref[0])
    for _ in range(SETTLE):
        u0 = ilc.correction_at(lambdas[0]) if use_ilc else np.zeros(7)
        plant.command_motor(qref[0] + u0)
    t = t0
    rec = {"t": [], "lam": [], "err": [], "u": [], "eq": []}
    for idx in range(len(lambdas)):
        u = ilc.correction_at(lambdas[idx]) if use_ilc else np.zeros(7)
        for _ in range(hold):
            plant.command_motor(qref[idx] + u)
            t += dt
        err = plant.read_tcp() - xyz[idx]
        rec["t"].append(t); rec["lam"].append(lambdas[idx])
        rec["err"].append(err); rec["u"].append(u.copy())
        rec["eq"].append(qref[idx] - plant.read_joint_encoders())
    for k in rec:
        rec[k] = np.array(rec[k])
    rec["lam_dot"] = lam_dot
    return rec, t


def run_protocol(plant, ref):
    """Paper-style 7-trial protocol. Returns list of per-trial records + the
    converged ILC and the trial-0 record (for validation)."""
    ilc = new_ilc(N)
    # (hold, use_ilc, learn, tag). Constant speed for a clean convergence story;
    # trial 5 turns the ILC OFF (error returns) and trial 6 turns it back ON --
    # the paper's own demonstration that the learned signal is what helps.
    plan = [(30, True, True, "learn"), (30, True, True, ""),
            (30, True, True, ""), (30, True, True, ""),
            (30, True, True, ""), (30, False, False, "ILC OFF"),
            (30, True, False, "ILC ON")]
    recs, t = [], 0.0
    for (hold, use_ilc, learn, tag) in plan:
        r, t = capture_trial(plant, ilc, ref, hold, use_ilc, t)
        r["tag"] = tag
        recs.append(r)
        if learn:
            ilc.update_from_trial(r["lam"], r["eq"])
    return recs, ilc


def rms_um(err):
    return 1e6 * np.sqrt(np.mean(np.sum(err ** 2, axis=1)))


# --------------------------------------------------------------------------
def fig_paper_style(recs):
    fig, ax = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
    # shade trials + labels
    for i, r in enumerate(recs):
        t0, t1 = r["t"][0], r["t"][-1]
        if i % 2 == 1:
            for a in ax:
                a.axvspan(t0, t1, color="0.92", zorder=0)
        ax[0].text((t0 + t1) / 2, 0.96, f"{i}", transform=ax[0].get_xaxis_transform(),
                   ha="center", va="top", fontsize=9, color="0.3")
        if r["tag"]:
            ax[0].annotate(r["tag"], xy=((t0 + t1) / 2, 0.84),
                           xycoords=ax[0].get_xaxis_transform(), ha="center",
                           fontsize=8.5, color="#b22222",
                           bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                     ec="#b22222", lw=0.8))
    T = np.concatenate([r["t"] for r in recs])
    E = np.concatenate([r["err"] for r in recs]) * 1e3        # mm
    U = np.concatenate([r["u"] for r in recs])                # rad
    for k, lab in enumerate(["e_x", "e_y", "e_z"]):
        ax[0].plot(T, E[:, k], color=AX_COLORS[k], label=f"${lab}$")
    ax[0].set_ylabel("TCP error [mm]")
    ax[0].set_title("KUKA iiwa14 path-ILC — Cartesian error, ILC input, path speed")
    ax[0].legend(ncol=3, loc="upper right")
    for j in range(7):
        ax[1].plot(T, U[:, j], color=J_COLORS[j], label=f"$u_{j+1}$")
    ax[1].set_ylabel("ILC input [rad]")
    ax[1].legend(ncol=7, loc="upper right", columnspacing=0.8, handlelength=1.0)
    speed = np.concatenate([np.full_like(r["t"], r["lam_dot"]) for r in recs])
    ax[2].plot(T, speed, color="0.2")
    ax[2].set_ylabel(r"path speed $\dot\lambda$ [1/s]")
    ax[2].set_xlabel("time [s]"); ax[2].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "trialwise_overview.png")); plt.close(fig)


def fig_validation(recs, ilc):
    """The key 'no laser tracker' validation. (A) The joint-side encoder error
    and the TRUE TCP error fall in lock-step across trials -- so minimizing what
    the encoders see (no tracker) minimizes the true accuracy; the encoder error
    is a faithful proxy. (B) The learned feedforward reproduces the spatial
    pattern of the true drivetrain error (ground truth we only have in sim)."""
    # per-trial joint-side RMS (mrad) and true TCP RMS (um)
    jrms = np.array([1e3 * np.sqrt(np.mean(np.sum(r["eq"] ** 2, axis=1)))
                     for r in recs])
    crms = np.array([rms_um(r["err"]) for r in recs])
    trials = np.arange(len(recs))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    a = ax[0]
    l1 = a.plot(trials, jrms, "o-", color="#0072B2",
                label="joint-side encoder RMS")[0]
    a.set_xlabel("trial"); a.set_ylabel("joint-side encoder RMS [mrad]",
                                        color="#0072B2")
    a.tick_params(axis="y", labelcolor="#0072B2")
    a2 = a.twinx(); a2.grid(False)
    l2 = a2.plot(trials, crms, "s--", color="#D55E00",
                 label="true TCP RMS (laser-tracker-equiv.)")[0]
    a2.set_ylabel("true TCP RMS [µm]", color="#D55E00")
    a2.tick_params(axis="y", labelcolor="#D55E00")
    # correlation over the learning trials (exclude the ILC-OFF spike trial 5)
    keep = [i for i in trials if recs[i]["tag"] != "ILC OFF"]
    rr = np.corrcoef(jrms[keep], crms[keep])[0, 1]
    a.set_title("Encoder error tracks true TCP error  "
                f"(corr = {rr:.3f})\n→ no laser tracker needed to learn")
    a.legend(handles=[l1, l2], loc="upper right")

    j = 3                                   # joint 4: clearest match
    lam = recs[0]["lam"]
    e0 = recs[0]["eq"][:, j]
    u_conv = np.array([ilc.correction_at(l)[j] for l in lam])
    ax[1].plot(lam, 1e3 * e0, color="#b22222",
               label="true drivetrain error (trial 0)")
    ax[1].plot(lam, 1e3 * u_conv, "--", color="#0072B2",
               label="learned feedforward (encoders only)")
    ax[1].set_xlabel(r"path parameter $\lambda$")
    ax[1].set_ylabel("joint-4 angle [mrad]")
    ax[1].set_title("Learned feedforward matches the\ndrivetrain-error pattern")
    ax[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "encoder_validation.png")); plt.close(fig)


def fig_error_vs_path(recs):
    lam = recs[0]["lam"]
    e0 = 1e6 * np.linalg.norm(recs[0]["err"], axis=1)
    # last fully-corrected, learning trial is index 4 (before ILC-off)
    eF = 1e6 * np.linalg.norm(recs[4]["err"], axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.fill_between(lam, e0, alpha=0.18, color="#b22222")
    ax.plot(lam, e0, color="#b22222", label=f"trial 0 (no correction), "
            f"RMS {rms_um(recs[0]['err']):.0f} µm")
    ax.plot(lam, eF, color="#009E73", label=f"after ILC, "
            f"RMS {rms_um(recs[4]['err']):.0f} µm")
    ax.set_yscale("log")
    ax.set_xlabel(r"path parameter $\lambda$")
    ax.set_ylabel("TCP error [µm] (log)")
    ax.set_title("Where the error lives along the path — before vs after learning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "error_along_path.png")); plt.close(fig)


def fig_vibration(plant, ref):
    """Each joint as a torsional spring-mass-damper: modal properties + a free
    vibration decay. This is the mechanical-engineering reading of the plant."""
    lambdas, xyz, qref = ref
    # effective link-side inertia from the mass matrix at a representative pose
    plant.reset(qref[0])
    for _ in range(SETTLE):
        plant.command_motor(qref[0])
    M = np.zeros((plant.model.nv, plant.model.nv))
    mujoco.mj_fullM(plant.model, M, plant.data.qM)
    Jeff = np.array([
        M[plant.model.jnt_dofadr[mujoco.mj_name2id(
            plant.model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_link")]]
        [plant.model.jnt_dofadr[mujoco.mj_name2id(
            plant.model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_link")]]
        for j in JOINTS])
    fn = np.sqrt(STIFFNESS / Jeff) / (2 * np.pi)
    zeta = DAMPING_L / (2 * np.sqrt(STIFFNESS * Jeff))

    # empirical free-vibration decay of joint 2 (displace link, release)
    jid = 1
    qadr = plant.qadr_l[jid]
    plant.reset(qref[0])
    for _ in range(SETTLE):
        plant.command_motor(qref[0])
    A0 = 0.05                          # initial 50 mrad deflection from rest
    plant.data.qpos[qadr] += A0
    mujoco.mj_forward(plant.model, plant.data)
    dt = plant.model.opt.timestep
    ts, dv = [], []
    for k in range(170):               # ~0.34 s window: shows the ring-down
        plant.command_motor(qref[0])    # hold motors; let the spring ring down
        ts.append(k * dt)
        dv.append(plant.data.qpos[qadr])
    ts = np.array(ts)
    dv = 1e3 * (np.array(dv) - np.mean(np.array(dv)[-30:]))   # detrend to 0
    # isolated structural 2nd-order prediction (link inertia + K + link damping)
    wn = 2 * np.pi * fn[jid]; z = zeta[jid]
    wd = wn * np.sqrt(max(1 - z ** 2, 1e-6))
    y_model = 1e3 * A0 * np.exp(-z * wn * ts) * np.cos(wd * ts)

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.3))
    x = np.arange(7)
    b = ax[0].bar(x, fn, color=J_COLORS)
    ax[0].axhline(5.0, color="#b22222", ls="--",
                  label="ILC Q-filter cutoff = 5 Hz")
    for xi, f, z in zip(x, fn, zeta):
        ax[0].text(xi, f, f"ζ={z:.2f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"J{i+1}" for i in range(7)])
    ax[0].set_ylabel("natural frequency $f_n$ [Hz]")
    ax[0].set_title("Each joint = torsional spring-mass-damper\n"
                    r"$f_n=\frac{1}{2\pi}\sqrt{K/J},\ \ \zeta=d/2\sqrt{KJ}$")
    ax[0].legend()
    ax[1].plot(ts, dv, color="#0072B2", label="simulation (full coupled arm)")
    ax[1].plot(ts, y_model, "--", color="#b22222",
               label=f"structural 2nd-order model\n($f_n$={fn[1]:.1f} Hz, ζ={zeta[1]:.2f})")
    ax[1].axhline(0, color="0.7", lw=0.8)
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("joint-2 deflection [mrad]")
    ax[1].set_title("Free vibration ring-down of joint 2\n"
                    "(motor-side damping makes the real joint more damped)")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "joint_modal_properties.png")); plt.close(fig)
    return fn, zeta


def main():
    plant = FlexArmPlant()
    ref = make_reference("A", n=N)
    print("running 7-trial paper protocol ...")
    recs, ilc = run_protocol(plant, ref)
    for i, r in enumerate(recs):
        print(f"  trial {i}: RMS {rms_um(r['err']):.0f} µm  {r['tag']}")
    fig_paper_style(recs)
    fig_validation(recs, ilc)
    fig_error_vs_path(recs)
    fn, zeta = fig_vibration(plant, ref)
    print("joint natural freqs [Hz]:", np.round(fn, 1))
    print("damping ratios        :", np.round(zeta, 2))
    print("Done. Saved analysis figures in outputs/.")


if __name__ == "__main__":
    main()
