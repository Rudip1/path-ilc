"""
Run all experiments and save figures.

  python experiments_toy.py

Produces in ../outputs:
  toy_convergence.png   - ILC convergence on the base path (the paper's core)
  toy_speed_generalization.png         - reusing the learned table at other speeds
  toy_path_transfer.png      - the open problem: a learned table on a NEW path
  toy_nn_path_transfer.png   - the neural-network layer predicts a correction for an unseen path

Everything is honest about its limits; see the README.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_arm import ArmPlant
from toy_simulation import make_reference, make_reference_morph, run_trial, new_ilc
import learned_correction
from plot_style import use_style, C, bar_log

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

SPEED = 0.4
SETTLE = 150
N_TRIALS = 7


def train_ilc_on(plant, kind="A", shift=(0, 0), scale=1.0, n=400,
                 n_trials=N_TRIALS):
    """Run ILC to convergence on one path; return (ilc, ref, history)."""
    lambdas, xz, qref = make_reference(kind, shift, scale, n=n)
    ilc = new_ilc(n)
    rms_hist, max_hist = [], []
    last = None
    for i in range(n_trials):
        r = run_trial(plant, ilc, lambdas, xz, qref,
                      speed=SPEED, settle=SETTLE, use_ilc=(i > 0))
        rms_hist.append(r["rms_um"]); max_hist.append(r["max_um"])
        ilc.update_from_trial(r["lambdas"], r["eq"])
        last = r
    return ilc, (lambdas, xz, qref), np.array(rms_hist), np.array(max_hist), last


def train_ilc_on_morph(plant, m, n=400, n_trials=N_TRIALS):
    """Run ILC to convergence on a morphed path (shape parameter m)."""
    lambdas, xz, qref = make_reference_morph(m, n=n)
    ilc = new_ilc(n)
    last = None
    for i in range(n_trials):
        r = run_trial(plant, ilc, lambdas, xz, qref,
                      speed=SPEED, settle=SETTLE, use_ilc=(i > 0))
        ilc.update_from_trial(r["lambdas"], r["eq"])
        last = r
    return ilc, (lambdas, xz, qref), None, None, last


def main():
    plant = ArmPlant()

    # ---- Goal: classical convergence on base path A --------------------
    print("[1/4] ILC convergence on base path A ...")
    ilcA, refA, rmsA, maxA, lastA = train_ilc_on(plant, "A")
    lambdas, xzA, qrefA = refA
    print(f"      RMS: {rmsA[0]:.0f} -> {rmsA[-1]:.0f} um "
          f"({rmsA[0]/rmsA[-1]:.1f}x)")

    # trial-0 (no ILC) and final trajectories for the path plot
    r0 = run_trial(plant, ilcA, lambdas, xzA, qrefA, speed=SPEED,
                   settle=SETTLE, use_ilc=False)
    rF = run_trial(plant, ilcA, lambdas, xzA, qrefA, speed=SPEED,
                   settle=SETTLE, use_ilc=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(range(len(rmsA)), rmsA, "o-", label="RMS")
    ax[0].plot(range(len(maxA)), maxA, "s--", label="max", alpha=0.7)
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("Cartesian error [um]")
    ax[0].set_title("ILC convergence (base path)"); ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(xzA[:, 0], xzA[:, 1], "k-", lw=2, label="desired")
    ax[1].plot(r0["xz_true"][:, 0], r0["xz_true"][:, 1], "b.", ms=3,
               alpha=0.5, label="trial 0 (no ILC)")
    ax[1].plot(rF["xz_true"][:, 0], rF["xz_true"][:, 1], "g.", ms=3,
               alpha=0.7, label="after ILC")
    ax[1].set_aspect("equal"); ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("z [m]")
    ax[1].set_title("Path tracking"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "toy_convergence.png"), dpi=130)
    plt.close(fig)

    # ---- Goal: speed generalization ------------------------------------
    print("[2/4] Speed generalization (reuse learned table) ...")
    speeds = [0.3, 0.4, 0.6]
    rms_by_speed = {}
    for sp in speeds:
        r = run_trial(plant, ilcA, lambdas, xzA, qrefA, speed=sp,
                      settle=SETTLE, use_ilc=True)
        rms_by_speed[sp] = r["rms_um"]
    r_nob = run_trial(plant, ilcA, lambdas, xzA, qrefA, speed=SPEED,
                      settle=SETTLE, use_ilc=False)
    for sp in speeds:
        tag = " (trained)" if abs(sp - SPEED) < 1e-9 else ""
        print(f"      speed {sp}: {rms_by_speed[sp]:.0f} um{tag}")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = [f"{s}x" + ("\n(trained)" if abs(s - SPEED) < 1e-9 else "")
              for s in speeds]
    vals = [rms_by_speed[s] for s in speeds]
    colors = [C.LEARNED if abs(s - SPEED) < 1e-9 else C.BASE for s in speeds]
    bar_log(ax, labels, vals, colors, ylabel="Cartesian RMS error [µm] (log)",
            title="Reusing the learned table at different speeds",
            annotate_reduction=False)
    ax.axhline(r_nob["rms_um"], color=C.WARM, ls="--", lw=1.6,
               label=f"no ILC ({r_nob['rms_um']:.0f} µm)")
    ax.set_ylim(top=max(ax.get_ylim()[1], r_nob["rms_um"] * 1.6))
    ax.legend(loc="upper center")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "toy_speed_generalization.png"))
    plt.close(fig)

    # ---- Goal: transfer to a DIFFERENT path (the open problem) ---------
    print("[3/4] Transfer to a new path B (open problem) ...")
    lamB, xzB, qrefB = make_reference("B", n=400)
    # (a) naive transfer: apply A's learned table on path B, zero new trials
    ilc_naive = new_ilc(len(lamB)); ilc_naive.load_table(ilcA.correction_table())
    r_naive = run_trial(plant, ilc_naive, lamB, xzB, qrefB, speed=SPEED,
                        settle=SETTLE, use_ilc=True)
    # (b) no correction baseline on B
    r_noneB = run_trial(plant, ilc_naive, lamB, xzB, qrefB, speed=SPEED,
                        settle=SETTLE, use_ilc=False)
    # (c) full relearn on B (the expensive option)
    ilcB, _, rmsB, _, _ = train_ilc_on(plant, "B")
    print(f"      no correction:     {r_noneB['rms_um']:.0f} um")
    print(f"      naive A->B:        {r_naive['rms_um']:.0f} um")
    print(f"      relearn on B:      {rmsB[-1]:.0f} um  ({len(rmsB)} trials)")

    # ---- Goal: neural-network layer predicts a correction for an unseen path
    print("[4/4] neural-network layer: predict correction for unseen path ...")
    if not learned_correction._HAS_TORCH:
        print("      torch not available; skipping neural-network layer.")
    else:
        # Build a training set that spans real SHAPE variation, not just tiny
        # shifts -- otherwise naive transfer already solves it and there is
        # nothing for the network to add. We interpolate the path shape with a
        # 'morph' parameter m in [0,1]: m=0 is path A, m=1 is path B-like, and
        # the network sees the geometry so it can learn the m-dependence.
        feats, tables = [], []
        rng = np.random.default_rng(0)
        morphs_train = list(np.linspace(0.0, 1.0, 10)) + \
            list(rng.uniform(0, 1, 6))
        for m in morphs_train:
            ilc_k, ref_k, _, _, _ = train_ilc_on_morph(plant, m, n_trials=5)
            lam_k, xz_k, _ = ref_k
            feats.append(learned_correction.path_features(lam_k, xz_k))
            tables.append(ilc_k.correction_table().reshape(-1))
        feats = np.array(feats); tables = np.array(tables)
        net, norm = learned_correction.train_correction_net(feats, tables, epochs=800)

        # unseen test: a morph value not in the training grid, where the shape
        # differs enough from A that naive A-transfer is poor
        test_m = 0.72
        lam_t, xz_t, qref_t = make_reference_morph(test_m, n=400)
        feat_t = learned_correction.path_features(lam_t, xz_t)
        pred = learned_correction.predict_table(net, norm, feat_t, N=len(lam_t))

        ilc_ai = new_ilc(len(lam_t)); ilc_ai.load_table(pred)
        r_ai = run_trial(plant, ilc_ai, lam_t, xz_t, qref_t, speed=SPEED,
                         settle=SETTLE, use_ilc=True)
        # baselines on the same unseen path
        r_none_t = run_trial(plant, ilc_ai, lam_t, xz_t, qref_t, speed=SPEED,
                             settle=SETTLE, use_ilc=False)
        # naive = reuse the m=0 (shape A) table, which is what you'd have if
        # you only ever trained on the base path
        ilcA0, refA0, _, _, _ = train_ilc_on_morph(plant, 0.0, n_trials=5)
        ilc_naive2 = new_ilc(len(lam_t))
        ilc_naive2.load_table(ilcA0.correction_table())
        r_naive2 = run_trial(plant, ilc_naive2, lam_t, xz_t, qref_t,
                             speed=SPEED, settle=SETTLE, use_ilc=True)
        print(f"      unseen path  no correction: {r_none_t['rms_um']:.0f} um")
        print(f"      unseen path  naive (A table): {r_naive2['rms_um']:.0f} um")
        print(f"      unseen path  NN (0 trials):   {r_ai['rms_um']:.0f} um")

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        names = ["no\ncorrection", "naive\n(A table)", "neural net\n(0 trials)"]
        vals = [r_none_t["rms_um"], r_naive2["rms_um"], r_ai["rms_um"]]
        bar_log(ax, names, vals, [C.NO_CORR, C.NAIVE, C.LEARNED],
                ylabel="Cartesian RMS error [µm] (log)",
                title="Neural-network correction on an unseen path (zero trials)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "toy_nn_path_transfer.png"))
        plt.close(fig)

    # transfer figure (built after we have relearn history)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    names = ["no\ncorrection", "naive\nA→B", "relearn B\n(7 trials)"]
    vals = [r_noneB["rms_um"], r_naive["rms_um"], rmsB[-1]]
    bar_log(ax, names, vals, [C.NO_CORR, C.NAIVE, C.BASE],
            ylabel="Cartesian RMS error [µm] (log)",
            title="Transfer to a different path (the open problem)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "toy_path_transfer.png"))
    plt.close(fig)

    print("\nDone. Figures saved in outputs/.")


if __name__ == "__main__":
    main()
