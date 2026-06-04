"""
Trial execution and the experiments for the three goals.

A "trial" = drive the arm once along a Cartesian path. At each path parameter
lambda in [0,1] the controller:
   1. looks up the desired TCP point gamma(lambda),
   2. computes reference joints qr via IDEAL inverse kinematics,
   3. adds the current ILC joint correction u[lambda],
   4. commands qr + u to the plant and steps physics,
   5. records the joint-encoder error (qr - q_measured) for the ILC, and the
      TRUE Cartesian error (for evaluation only).

Goal mapping:
  exp_convergence       -> classical ILC convergence on one path
  exp_speed             -> reuse learned table at a different speed (no relearn)
  exp_transfer          -> reuse learned table on a DIFFERENT path (the open
                           problem); naive transfer vs. fresh relearn
The learned layer (learned_correction.py) consumes tables produced here.
"""

import numpy as np
from toy_arm import ArmPlant, inverse_kinematics
from ilc import PathILC

# ---- path library --------------------------------------------------------
# Paths live in the arm's x-z plane, kept well inside the workspace.


def path_point(lam, kind="A", shift=(0.0, 0.0), scale=1.0):
    """Return desired TCP (x, z) at path parameter lam in [0,1].

    'A' is the base path the ILC trains on. Variants change geometry so the
    transfer experiment is meaningful: 'shift' translates, 'scale' changes
    amplitude, 'B'/'C' change the shape (frequency / curvature)."""
    cx, cz = 0.55, 0.15  # path center
    if kind == "A":
        x = cx + 0.18 * np.cos(2 * np.pi * lam)
        z = cz + 0.12 * np.sin(2 * np.pi * lam)
    elif kind == "B":          # different shape: figure-eight-ish
        x = cx + 0.18 * np.cos(2 * np.pi * lam)
        z = cz + 0.12 * np.sin(4 * np.pi * lam)
    elif kind == "C":          # different curvature + tilt
        x = cx + 0.16 * np.cos(2 * np.pi * lam + 0.4)
        z = cz + 0.14 * np.sin(2 * np.pi * lam) + 0.03 * np.cos(6 * np.pi * lam)
    else:
        raise ValueError(kind)
    return np.array([x * scale + shift[0], z * scale + shift[1]])


def path_point_morph(lam, m):
    """A path whose SHAPE morphs with m in [0,1]: m=0 -> shape A (single
    sine in z), m=1 -> shape B (double-frequency sine in z). Intermediate m
    blends the two. The geometry changes continuously, so naive transfer of a
    single learned table degrades as m moves away from where it was learned --
    which is exactly the regime where a geometry-aware predictor can help."""
    cx, cz = 0.55, 0.15
    x = cx + 0.18 * np.cos(2 * np.pi * lam)
    z_A = 0.12 * np.sin(2 * np.pi * lam)
    z_B = 0.12 * np.sin(4 * np.pi * lam)
    z = cz + (1 - m) * z_A + m * z_B
    return np.array([x, z])


def make_reference_morph(m, n=400):
    """Reference for a morphed path (see path_point_morph)."""
    lambdas = np.linspace(0.0, 1.0, n)
    xz = np.array([path_point_morph(l, m) for l in lambdas])
    q_ref = np.zeros((n, 3))
    q_prev = None
    for i, p in enumerate(xz):
        q_prev = inverse_kinematics(p, q_init=q_prev)
        q_ref[i] = q_prev
    return lambdas, xz, q_ref


def make_reference(kind="A", shift=(0.0, 0.0), scale=1.0, n=400):
    """Precompute the desired Cartesian points and ideal reference joints
    along a path. Returns (lambdas, xz_ref, q_ref)."""
    lambdas = np.linspace(0.0, 1.0, n)
    xz = np.array([path_point(l, kind, shift, scale) for l in lambdas])
    q_ref = np.zeros((n, 3))
    q_prev = None
    for i, p in enumerate(xz):
        q_prev = inverse_kinematics(p, q_init=q_prev)
        q_ref[i] = q_prev
    return lambdas, xz, q_ref


# ---- single trial --------------------------------------------------------

def run_trial(plant, ilc, lambdas, xz_ref, q_ref, speed=1.0,
              settle=60, use_ilc=True):
    """Execute one trial. 'speed' resamples how fast lambda advances to test
    speed generalization (the ILC table is indexed by lambda, so it should
    transfer). Returns dict with joint error (for ILC) and true Cartesian
    error (for evaluation)."""
    n = len(lambdas)
    # resample the lambda schedule for the requested speed: fewer control
    # steps when faster, more when slower. We pick visited indices accordingly.
    n_steps = max(int(round(n / speed)), 10)
    visit = np.linspace(0, n - 1, n_steps).round().astype(int)

    plant.reset(q_ref[0])
    # let the controller settle at the start so trial 0 isn't dominated by
    # the initial transient
    for _ in range(settle):
        u0 = ilc.correction_at(lambdas[0]) if use_ilc else np.zeros(3)
        plant.command_joint(q_ref[0] + u0)

    lam_hist, eq_hist, xz_true, xz_des = [], [], [], []
    for idx in visit:
        qr = q_ref[idx]
        u = ilc.correction_at(lambdas[idx]) if use_ilc else np.zeros(3)
        plant.command_joint(qr + u)
        q_meas = plant.read_encoders()
        lam_hist.append(lambdas[idx])
        eq_hist.append(qr - q_meas)        # joint-encoder error -> ILC
        xz_true.append(plant.read_tcp())   # ground truth -> evaluation only
        xz_des.append(xz_ref[idx])

    xz_true = np.array(xz_true)
    xz_des = np.array(xz_des)
    cart_err = np.linalg.norm(xz_true - xz_des, axis=1)  # meters
    return {
        "lambdas": np.array(lam_hist),
        "eq": np.array(eq_hist),
        "cart_err_m": cart_err,
        "rms_um": 1e6 * np.sqrt(np.mean(cart_err ** 2)),
        "max_um": 1e6 * np.max(cart_err),
        "xz_true": xz_true,
        "xz_des": xz_des,
    }


def new_ilc(n):
    """Fresh ILC sized to the reference length, with the paper's gains."""
    # Nq scaled down from the paper's 45 (their N=10000) to suit our N~400.
    return PathILC(n_intervals=n, n_joints=3, Kp_ilc=1.0, Kd_ilc=0.01,
                   Nq=8, f_cutoff_hz=5.0, Ts=1e-3)
