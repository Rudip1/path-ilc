"""
The four experiments on the FLEXIBLE-JOINT KUKA iiwa14.

    python src/experiments_kuka.py

Produces in ../outputs:
  kuka_convergence.png  - ILC convergence on a 3D base path (joint-side encoders only)
  kuka_speed_generalization.png        - reusing the learned table at different traversal speeds
  kuka_path_transfer.png     - the open problem: a learned table on a NEW 3D path
  kuka_nn_path_transfer.png  - the neural-network layer predicts a correction for an unseen path

Same algorithm as the toy demo, now on a real 7-DOF industrial arm whose hidden
error is genuine joint elasticity / transmission deflection (see kuka_plant.py).
All errors are Cartesian RMS at the TCP, in micrometers (um).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from kuka_plant import FlexArmPlant, Effects   # noqa: E402
from kuka_simulation import (make_reference, make_reference_morph, run_trial,  # noqa: E402
                      new_ilc, _CENTER)
import learned_correction  # noqa: E402
from plot_style import use_style, C, bar_log  # noqa: E402

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

N = 200
HOLD = 30
SETTLE = 500
N_TRIALS = 7
# headline experiments run under the realistic (repeatable) drivetrain effects:
# transmission error + Stribeck friction + backlash + encoder noise. Thermal
# drift is non-repetitive and is studied separately (figures_realism / dashboards).
HEADLINE = Effects.realistic_repeatable()


def train_ilc(plant, kind="A", n_trials=N_TRIALS):
    lambdas, xyz, qref = make_reference(kind, n=N)
    ilc = new_ilc(len(lambdas))
    rms_hist, max_hist, last = [], [], None
    for i in range(n_trials):
        r = run_trial(plant, ilc, lambdas, xyz, qref,
                      hold=HOLD, settle=SETTLE, use_ilc=(i > 0))
        rms_hist.append(r["rms_um"]); max_hist.append(r["max_um"])
        ilc.update_from_trial(r["lambdas"], r["eq"])
        last = r
    return ilc, (lambdas, xyz, qref), np.array(rms_hist), np.array(max_hist), last


def train_ilc_morph(plant, m, n_trials=5):
    lambdas, xyz, qref = make_reference_morph(m, n=N)
    ilc = new_ilc(len(lambdas))
    for i in range(n_trials):
        r = run_trial(plant, ilc, lambdas, xyz, qref,
                      hold=HOLD, settle=SETTLE, use_ilc=(i > 0))
        ilc.update_from_trial(r["lambdas"], r["eq"])
    return ilc, (lambdas, xyz, qref)


def fig_convergence(rmsA, maxA, xyzA, r0, rF):
    """Two balanced panels: trial-wise convergence (log) and 3D tool-path."""
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.16)
    ax0 = fig.add_subplot(gs[0])
    ax0.semilogy(range(len(rmsA)), rmsA, "o-", color=C.BASE, label="RMS")
    ax0.semilogy(range(len(maxA)), maxA, "s--", color=C.NAIVE, alpha=0.9,
                 label="max")
    ax0.set_xlabel("trial"); ax0.set_ylabel("Cartesian error [µm] (log)")
    ax0.set_title("ILC convergence (joint-side encoder only)"); ax0.legend()
    ax1 = fig.add_subplot(gs[1], projection="3d")
    ax1.plot(xyzA[:, 0], xyzA[:, 1], xyzA[:, 2], "-", color="0.15", lw=2.4,
             label="desired path")
    ax1.plot(r0["xyz_true"][:, 0], r0["xyz_true"][:, 1], r0["xyz_true"][:, 2],
             "o", color=C.WARM, ms=3.4, alpha=0.7, label="trial 0 (no ILC)")
    ax1.plot(rF["xyz_true"][:, 0], rF["xyz_true"][:, 1], rF["xyz_true"][:, 2],
             "o", color=C.LEARNED, ms=3.4, alpha=0.9, label="after ILC")
    # fill the cube: tight limits + zoomed box so the path is not a tiny blob
    for setlim, d in ((ax1.set_xlim, 0), (ax1.set_ylim, 1), (ax1.set_zlim, 2)):
        lo, hi = xyzA[:, d].min(), xyzA[:, d].max(); pad = 0.08 * (hi - lo)
        setlim(lo - pad, hi + pad)
    ax1.set_box_aspect((1, 1, 1), zoom=1.28)
    ax1.view_init(elev=22, azim=-58)
    ax1.set_xlabel("x [m]", labelpad=-4); ax1.set_ylabel("y [m]", labelpad=-4)
    ax1.set_zlabel("z [m]", labelpad=-4)
    ax1.tick_params(labelsize=7, pad=-1)
    for pane in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        pane.pane.set_facecolor("white"); pane.pane.set_alpha(0.6)
    ax1.set_title("3D tool-path tracking")
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9,
               handletextpad=0.4, borderpad=0.3)
    fig.savefig(os.path.join(OUT, "kuka_convergence.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    plant = FlexArmPlant(effects=HEADLINE)

    # ---- 1) convergence on the 3D base path ----------------------------
    print("[1/4] ILC convergence on KUKA base path (joint-side encoders only) ...")
    ilcA, refA, rmsA, maxA, lastA = train_ilc(plant, "A")
    lambdas, xyzA, qrefA = refA
    print(f"      RMS: {rmsA[0]:.0f} -> {rmsA[-1]:.0f} um  ({rmsA[0]/rmsA[-1]:.1f}x)")

    r0 = run_trial(plant, ilcA, lambdas, xyzA, qrefA, hold=HOLD,
                   settle=SETTLE, use_ilc=False)
    rF = run_trial(plant, ilcA, lambdas, xyzA, qrefA, hold=HOLD,
                   settle=SETTLE, use_ilc=True)
    fig_convergence(rmsA, maxA, xyzA, r0, rF)

    # ---- 2) speed generalization (reuse table at other speeds) ---------
    print("[2/4] Speed generalization (reuse learned table) ...")
    holds = [20, 30, 45]
    rms_by = {h: run_trial(plant, ilcA, lambdas, xyzA, qrefA, hold=h,
                           settle=SETTLE, use_ilc=True)["rms_um"] for h in holds}
    r_nob = run_trial(plant, ilcA, lambdas, xyzA, qrefA, hold=HOLD,
                      settle=SETTLE, use_ilc=False)
    for h in holds:
        tag = " (trained)" if h == HOLD else ""
        print(f"      hold {h} steps: {rms_by[h]:.0f} um{tag}")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = [f"{h}\n(trained)" if h == HOLD else f"{h}" for h in holds]
    vals = [rms_by[h] for h in holds]
    colors = [C.LEARNED if h == HOLD else C.BASE for h in holds]
    bar_log(ax, labels, vals, colors, ylabel="Cartesian RMS error [µm] (log)",
            title="Reusing the learned table at different speeds",
            annotate_reduction=False)
    ax.axhline(r_nob["rms_um"], color=C.WARM, ls="--", lw=1.6,
               label=f"no ILC ({r_nob['rms_um']:.0f} µm)")
    ax.set_ylim(top=max(ax.get_ylim()[1], r_nob["rms_um"] * 1.6))
    ax.set_xlabel("steps held per path point (slower →)")
    ax.legend(loc="upper center")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "kuka_speed_generalization.png"))
    plt.close(fig)

    # ---- 3) transfer to a different 3D path (the open problem) ---------
    print("[3/4] Transfer to a new path B (open problem) ...")
    lamB, xyzB, qrefB = make_reference("B", n=N)
    ilc_naive = new_ilc(len(lamB)); ilc_naive.load_table(ilcA.correction_table())
    r_naive = run_trial(plant, ilc_naive, lamB, xyzB, qrefB, hold=HOLD,
                        settle=SETTLE, use_ilc=True)
    r_noneB = run_trial(plant, ilc_naive, lamB, xyzB, qrefB, hold=HOLD,
                        settle=SETTLE, use_ilc=False)
    ilcB, _, rmsB, _, _ = train_ilc(plant, "B")
    print(f"      no correction: {r_noneB['rms_um']:.0f} um")
    print(f"      naive A->B:    {r_naive['rms_um']:.0f} um")
    print(f"      relearn on B:  {rmsB[-1]:.0f} um  ({len(rmsB)} trials)")

    # ---- 4) neural-network layer: predict a correction for an unseen path --
    print("[4/4] neural-network layer: predict correction for unseen path ...")
    nn_vals = None
    if learned_correction._HAS_TORCH:
        feats, tables = [], []
        rng = np.random.default_rng(0)
        morphs = list(np.linspace(0.0, 1.0, 9)) + list(rng.uniform(0, 1, 5))
        for m in morphs:
            ilc_m, ref_m = train_ilc_morph(plant, m, n_trials=5)
            lam_m, xyz_m, _ = ref_m
            feats.append(learned_correction.path_features(lam_m, xyz_m))
            tables.append(ilc_m.correction_table().reshape(-1))
        net, norm = learned_correction.train_correction_net(np.array(feats), np.array(tables),
                                            epochs=800)
        test_m = 0.72
        lam_t, xyz_t, qref_t = make_reference_morph(test_m, n=N)
        pred = learned_correction.predict_table(net, norm, learned_correction.path_features(lam_t, xyz_t),
                                N=len(lam_t), dof=7)
        ilc_ai = new_ilc(len(lam_t)); ilc_ai.load_table(pred)
        r_ai = run_trial(plant, ilc_ai, lam_t, xyz_t, qref_t, hold=HOLD,
                         settle=SETTLE, use_ilc=True)
        r_none_t = run_trial(plant, ilc_ai, lam_t, xyz_t, qref_t, hold=HOLD,
                             settle=SETTLE, use_ilc=False)
        ilcA0, _ = train_ilc_morph(plant, 0.0, n_trials=5)
        ilc_nv = new_ilc(len(lam_t)); ilc_nv.load_table(ilcA0.correction_table())
        r_nv = run_trial(plant, ilc_nv, lam_t, xyz_t, qref_t, hold=HOLD,
                         settle=SETTLE, use_ilc=True)
        nn_vals = [r_none_t["rms_um"], r_nv["rms_um"], r_ai["rms_um"]]
        print(f"      unseen path  no correction: {nn_vals[0]:.0f} um")
        print(f"      unseen path  naive (A table): {nn_vals[1]:.0f} um")
        print(f"      unseen path  NN (0 trials):   {nn_vals[2]:.0f} um")

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        names = ["no\ncorrection", "naive\n(base table)", "neural net\n(0 trials)"]
        bar_log(ax, names, nn_vals, [C.NO_CORR, C.NAIVE, C.LEARNED],
                ylabel="Cartesian RMS error [µm] (log)",
                title="Neural-network correction on an unseen 3D path (zero trials)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "kuka_nn_path_transfer.png"))
        plt.close(fig)
    else:
        print("      torch not available; skipping neural-network layer.")

    # transfer figure
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    names = ["no\ncorrection", "naive\nA→B", f"relearn B\n({len(rmsB)} trials)"]
    vals = [r_noneB["rms_um"], r_naive["rms_um"], rmsB[-1]]
    bar_log(ax, names, vals, [C.NO_CORR, C.NAIVE, C.BASE],
            ylabel="Cartesian RMS error [µm] (log)",
            title="Transfer to a different 3D path (the open problem)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "kuka_path_transfer.png"))
    plt.close(fig)

    print("\nDone. Figures saved in outputs/.")


if __name__ == "__main__":
    main()
