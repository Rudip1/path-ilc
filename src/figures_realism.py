"""
Figures for the REALISTIC drivetrain effects (transmission error, Stribeck
friction, backlash, encoder noise, thermal drift). See kuka.Effects.

    python src/figures_realism.py

Writes to ../outputs:
  realistic_effects_ablation.png      - ILC stays accurate under each realistic effect
                         (repeatable effects are learned; noise is filtered)
  qfilter_noise_rejection.png       - the Gaussian Q-filter rejects encoder noise: without
                         it the ILC learns the noise and the correction is junk
  transmission_error_profile.png  - the injected angle-periodic transmission-error profile
  thermal_frozen_vs_online.png       - thermal drift defeats a FROZEN ILC table, while an
                         ONLINE (continuously updating) ILC tracks the warm-up
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("MUJOCO_GL", "glx")
sys.path.insert(0, os.path.dirname(__file__))
from kuka_plant import FlexArmPlant, Effects, STIFFNESS  # noqa: E402
from ilc import PathILC  # noqa: E402
from kuka_simulation import make_reference, run_trial  # noqa: E402
from plot_style import use_style, C, label_bars  # noqa: E402

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

N, HOLD, SETTLE = 200, 30, 500
REF = None


def ilc_of(nq=6):
    return PathILC(n_intervals=N, n_joints=7, Kp_ilc=1.0, Kd_ilc=0.01,
                   Nq=nq, f_cutoff_hz=5.0, Ts=2e-3)


def run_session(effects, n_trials=7, nq=6, freeze_after=None):
    """Run an ILC session; return (rms per trial, ilc, plant)."""
    lambdas, xyz, qref = REF
    plant = FlexArmPlant(effects=effects)
    ilc = ilc_of(nq)
    rms = []
    for i in range(n_trials):
        r = run_trial(plant, ilc, lambdas, xyz, qref, hold=HOLD,
                      settle=SETTLE, use_ilc=(i > 0))
        rms.append(r["rms_um"])
        if freeze_after is None or i < freeze_after:
            ilc.update_from_trial(r["lambdas"], r["eq"])
    return np.array(rms), ilc, plant


# --------------------------------------------------------------------------
def fig_ablation():
    configs = [
        ("baseline", Effects()),
        ("+transm.\nerror", Effects(te=True)),
        ("+Stribeck\nfriction", Effects(stribeck=True)),
        ("+backlash", Effects(backlash=True)),
        ("+encoder\nnoise", Effects(encoder_noise=True)),
        ("+thermal", Effects(thermal=True)),
        ("ALL\nrealistic", Effects.realistic()),
    ]
    raw, final = [], []
    for name, fx in configs:
        rms, _, _ = run_session(fx, n_trials=7)
        raw.append(rms[0]); final.append(rms[-1])
        print(f"  {name.replace(chr(10),' '):22s} raw={rms[0]:7.0f}  final={rms[-1]:6.0f} um")
    x = np.arange(len(configs)); w = 0.4
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x - w / 2, raw, w, color=C.NO_CORR, zorder=3, edgecolor="white",
           linewidth=0.7, label="trial 0 (no correction)")
    ax.bar(x + w / 2, final, w, color=C.ILC, zorder=3, edgecolor="white",
           linewidth=0.7, label="after ILC")
    for xi, v in zip(x, final):
        ax.text(xi + w / 2, v * 1.04, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in configs])
    ax.set_ylabel("Cartesian RMS [µm] (log)")
    ax.set_title("Path-ILC stays accurate under realistic drivetrain effects\n"
                 "(repeatable effects are learned away; encoder noise is filtered)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "realistic_effects_ablation.png")); plt.close(fig)


def fig_qfilter():
    # stress-test noise level (coarser + larger than a good 19-bit encoder) so
    # the filter's effect is visible; the point is the mechanism, not the number.
    fx = Effects(encoder_noise=True, enc_sigma=6e-4, enc_bits=13)
    rms_f, ilc_f, _ = run_session(fx, n_trials=8, nq=6)    # with Q-filter
    rms_n, ilc_n, _ = run_session(fx, n_trials=8, nq=0)    # no Q-filter
    lam = REF[0]
    uf = np.array([ilc_f.correction_at(l)[1] for l in lam]) * 1e3
    un = np.array([ilc_n.correction_at(l)[1] for l in lam]) * 1e3
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax[0].plot(range(8), rms_n, "s--", color=C.WARM, label="no Q-filter")
    ax[0].plot(range(8), rms_f, "o-", color=C.LEARNED, label="Gaussian Q-filter")
    ax[0].set_yscale("log"); ax[0].set_xlabel("trial")
    ax[0].set_ylabel("Cartesian RMS [µm] (log)")
    ax[0].set_title("Convergence under noisy encoders"); ax[0].legend()
    ax[1].plot(lam, un, color=C.WARM, lw=1.0, label="no Q-filter (learns noise)")
    ax[1].plot(lam, uf, color=C.LEARNED, label="Gaussian Q-filter (smooth)")
    ax[1].set_xlabel(r"path parameter $\lambda$")
    ax[1].set_ylabel("learned correction, joint 2 [mrad]")
    ax[1].set_title("The Q-filter stops the ILC learning noise"); ax[1].legend()
    fig.suptitle("Gaussian Q-filter rejects encoder noise (its purpose in the paper)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "qfilter_noise_rejection.png")); plt.close(fig)


def fig_transmission():
    fx = Effects()           # defaults hold the TE harmonics
    th = np.linspace(-1.2, 1.2, 800)
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for j, c in zip([0, 2, 4], ["#0072B2", "#D55E00", "#009E73"]):
        eps = 1e3 * fx.te_amp[j] * np.sin(fx.te_n1[j] * th + fx.te_phase[j])
        ax.plot(th, eps, color=c, label=f"joint {j+1}  ({int(fx.te_n1[j])} cyc/rev)")
    ax.set_xlabel("joint angle θ [rad]")
    ax.set_ylabel("transmission error ε(θ) [mrad]")
    ax.set_title("Injected angle-periodic transmission error\n"
                 "(gear/cycloidal harmonics — 'up to 100× per revolution')")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "transmission_error_profile.png")); plt.close(fig)


def fig_thermal():
    n_trials, freeze = 14, 6
    # frozen-after-`freeze` table
    rms_fr, _, plant_fr = run_session(Effects(thermal=True), n_trials=n_trials,
                                      freeze_after=freeze)
    # need temperature trace: rerun frozen capturing temp
    lambdas, xyz, qref = REF
    plant = FlexArmPlant(effects=Effects(thermal=True))
    ilc = ilc_of()
    rms_frozen, temp_tr = [], []
    for i in range(n_trials):
        r = run_trial(plant, ilc, lambdas, xyz, qref, hold=HOLD, settle=SETTLE,
                      use_ilc=(i > 0))
        rms_frozen.append(r["rms_um"]); temp_tr.append(plant.read_temperature().max())
        if i < freeze:
            ilc.update_from_trial(r["lambdas"], r["eq"])
    # online (keeps updating)
    plant2 = FlexArmPlant(effects=Effects(thermal=True))
    ilc2 = ilc_of()
    rms_online = []
    for i in range(n_trials):
        r = run_trial(plant2, ilc2, lambdas, xyz, qref, hold=HOLD, settle=SETTLE,
                      use_ilc=(i > 0))
        rms_online.append(r["rms_um"]); ilc2.update_from_trial(r["lambdas"], r["eq"])

    t = np.arange(n_trials)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(t, rms_frozen, "o-", color=C.WARM,
            label=f"frozen ILC (table fixed after trial {freeze-1})")
    ax.plot(t, rms_online, "s-", color=C.LEARNED,
            label="online ILC (keeps updating)")
    ax.axvline(freeze - 1, color="0.6", ls=":", lw=1)
    ax.annotate("table frozen", xy=(freeze - 1, rms_frozen[freeze - 1]),
                xytext=(freeze + 0.3, rms_frozen[-1] * 1.3), fontsize=9,
                color="0.4")
    ax.set_yscale("log")
    ax.set_xlabel("trial"); ax.set_ylabel("Cartesian RMS [µm] (log)")
    ax.set_title("Thermal drift defeats a frozen correction;\n"
                 "an online (self-learning) ILC tracks the warm-up")
    ax.legend(loc="center left")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(t, temp_tr, color=C.NAIVE, alpha=0.7)
    ax2.set_ylabel("joint temperature (a.u.)", color=C.NAIVE)
    ax2.tick_params(axis="y", labelcolor=C.NAIVE)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "thermal_frozen_vs_online.png")); plt.close(fig)


TEMP_PROFILE = np.array([1.0, 1.0, 0.8, 0.8, 0.6, 0.5, 0.5])   # differential heating


def _converge_at_temp(level, n_trials=7):
    """Fresh plant held at a fixed thermal state; ILC converges; return table."""
    lambdas, xyz, qref = REF
    plant = FlexArmPlant(effects=Effects(thermal=True))
    plant.set_temperature(level * TEMP_PROFILE)
    ilc = ilc_of()
    last = None
    for i in range(n_trials):
        r = run_trial(plant, ilc, lambdas, xyz, qref, hold=HOLD, settle=SETTLE,
                      use_ilc=(i > 0))
        ilc.update_from_trial(r["lambdas"], r["eq"]); last = r
    return ilc.correction_table(), last["rms_um"]


def _eval_table_at_temp(level, table):
    """Apply a given correction table at a fixed thermal state; return RMS."""
    lambdas, xyz, qref = REF
    plant = FlexArmPlant(effects=Effects(thermal=True))
    plant.set_temperature(level * TEMP_PROFILE)
    ilc = ilc_of(); ilc.load_table(table)
    r = run_trial(plant, ilc, lambdas, xyz, qref, hold=HOLD, settle=SETTLE,
                  use_ilc=True)
    return r["rms_um"]


def fig_thermal_ai():
    """Temperature-aware learned layer (a temperature-indexed linear model): learn correction = f(joint temperature), then
    predict the correction for an UNSEEN thermal state with zero trials. Beats a
    frozen cold table; matches the oracle that re-learns at that temperature."""
    levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    test_level = 0.7
    tables, temps = [], []
    for L in levels:
        tab, _ = _converge_at_temp(L)
        tables.append(tab.reshape(-1)); temps.append(L * TEMP_PROFILE)
        print(f"  trained at temp level {L:.2f}")
    X = np.array(temps); Y = np.array(tables)
    Xb = np.hstack([X, np.ones((len(X), 1))])               # bias
    lam = 1e-6
    B = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)

    def predict(temp_vec):
        return (np.append(temp_vec, 1.0) @ B).reshape(N, 7)

    # held-out hot state
    frozen_rms = _eval_table_at_temp(test_level, tables[0].reshape(N, 7))  # cold table
    nn_table = predict(test_level * TEMP_PROFILE)
    nn_rms = _eval_table_at_temp(test_level, nn_table)
    oracle_table, oracle_rms = _converge_at_temp(test_level)
    print(f"  test temp {test_level}:  frozen(cold)={frozen_rms:.0f}  "
          f"learned(0 trials)={nn_rms:.0f}  oracle={oracle_rms:.0f} um")

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4))
    names = ["frozen\n(cold table)", "learned\n(0 trials)", "oracle\n(re-learn)"]
    vals = np.array([frozen_rms, nn_rms, oracle_rms], dtype=float)
    ax[0].bar(names, vals, color=[C.WARM, C.LEARNED, C.BASE], zorder=3,
              edgecolor="white", linewidth=0.7)
    ax[0].set_yscale("log")
    ax[0].set_ylim(10 ** np.floor(np.log10(vals.min() * 0.6)), vals.max() * 2.6)
    label_bars(ax[0], vals)
    ax[0].set_ylabel("Cartesian RMS [µm] (log)")
    ax[0].set_title(f"Correction at an unseen thermal state\n(temp level {test_level})")
    lam_arr = REF[0]
    ax[1].plot(lam_arr, 1e3 * oracle_table[:, 1], color=C.BASE,
               label="oracle (re-learned hot)")
    ax[1].plot(lam_arr, 1e3 * nn_table[:, 1], "--", color=C.LEARNED,
               label="learned (linear) from temperature")
    ax[1].plot(lam_arr, 1e3 * tables[0].reshape(N, 7)[:, 1], ":", color=C.WARM,
               label="cold table (frozen)")
    ax[1].set_xlabel(r"path parameter $\lambda$")
    ax[1].set_ylabel("joint-2 correction [mrad]")
    ax[1].set_title("Predicted vs required correction"); ax[1].legend(fontsize=8)
    fig.suptitle("Temperature-aware learned layer predicts the thermal-drift "
                 "compensation (a linear model on top of ILC)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "thermal_learned_compensation.png")); plt.close(fig)


def main():
    global REF
    REF = make_reference("A", n=N)
    print("[1/5] ablation across realistic effects ...")
    fig_ablation()
    print("[2/5] Q-filter noise rejection ...")
    fig_qfilter()
    print("[3/5] transmission-error profile ...")
    fig_transmission()
    print("[4/5] thermal drift: frozen vs online ILC ...")
    fig_thermal()
    print("[5/5] temperature-aware learned layer ...")
    fig_thermal_ai()
    print("Done. Saved realistic-effects figures in outputs/.")


if __name__ == "__main__":
    main()
