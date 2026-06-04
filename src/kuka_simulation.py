"""
3D path library, inverse kinematics, and the trial loop for the flexible KUKA.

A "trial" drives the arm once along a 3D Cartesian path. At each path parameter
lambda in [0,1] the controller:
   1. looks up the desired TCP point gamma(lambda),
   2. computes reference motor angles qr by IDEAL inverse kinematics on the
      RIGID model (it assumes link == motor command),
   3. adds the current ILC joint correction u[lambda],
   4. commands qr + u to the motors and steps the flexible plant,
   5. records the JOINT-SIDE encoder error (qr - q_link_abs) for the ILC, and
      the TRUE Cartesian TCP error (for evaluation only).

The ILC (ilc.PathILC) is the very same path-indexed PD law with Gaussian
Q-filter from the paper, now with n_joints = 7.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from kuka_plant import FlexArmPlant, RigidKinematics, HOME, JOINTS  # noqa: E402
from ilc import PathILC  # noqa: E402

# A shared rigid-kinematics helper for planning/IK (cheap, stateless-ish).
_KIN = RigidKinematics()
_CENTER = _KIN.fk(HOME)   # base path is centered on the home TCP


# ---- 3D path library ------------------------------------------------------

def path_point(lam, kind="A"):
    """Desired TCP (x,y,z) at path parameter lam in [0,1]. Curves live in a
    tilted plane around the home TCP, well inside the workspace."""
    c = _CENTER
    a = 2 * np.pi * lam
    if kind == "A":            # circle in a slightly tilted plane
        return c + np.array([0.12 * np.cos(a),
                             0.12 * np.sin(a),
                             0.05 * np.sin(a)])
    if kind == "B":            # different shape: vertical figure-eight
        return c + np.array([0.12 * np.cos(a),
                             0.10 * np.sin(2 * a),
                             0.10 * np.sin(a)])
    raise ValueError(kind)


def path_point_morph(lam, m):
    """Shape morphs with m in [0,1]: m=0 -> circle (A), m=1 -> figure-eight in
    the y-channel. Intermediate m blends them. Geometry changes continuously,
    so a single learned table degrades away from where it was learned -- the
    regime where a geometry-aware predictor helps."""
    c = _CENTER
    a = 2 * np.pi * lam
    y_A = 0.12 * np.sin(a)
    y_B = 0.10 * np.sin(2 * a)
    return c + np.array([0.12 * np.cos(a),
                         (1 - m) * y_A + m * y_B,
                         0.05 * np.sin(a)])


def make_reference(kind="A", n=300):
    """Precompute desired Cartesian points and ideal reference motor angles."""
    lambdas = np.linspace(0.0, 1.0, n)
    xyz = np.array([path_point(l, kind) for l in lambdas])
    return lambdas, xyz, _ik_chain(xyz)


def make_reference_morph(m, n=300):
    lambdas = np.linspace(0.0, 1.0, n)
    xyz = np.array([path_point_morph(l, m) for l in lambdas])
    return lambdas, xyz, _ik_chain(xyz)


def _ik_chain(xyz):
    """IK along a path, warm-starting each point from the previous solution."""
    q = np.zeros((len(xyz), 7))
    q_prev = HOME.copy()
    for i, p in enumerate(xyz):
        q_prev = _KIN.ik(p, q_init=q_prev)
        q[i] = q_prev
    return q


# ---- single trial ---------------------------------------------------------

def run_trial(plant, ilc, lambdas, xyz_ref, q_ref, hold=25, settle=400,
              use_ilc=True):
    """Execute one trial on the flexible plant.

    `hold` is the number of physics steps the controller dwells at each path
    point -- the path-parameter speed is ~1/hold. A larger `hold` means a
    slower, more quasi-static traversal (tighter tracking), so the residual
    error is dominated by the repeatable, pose-dependent transmission/elastic
    error that the path-indexed ILC is designed to learn. Speed generalization
    is tested by reusing the SAME learned table at a different `hold`.

    Returns joint-side error (for the ILC) and true Cartesian error (eval only).
    """
    n = len(lambdas)
    plant.reset(q_ref[0])
    for _ in range(settle):
        u0 = ilc.correction_at(lambdas[0]) if use_ilc else np.zeros(7)
        plant.command_motor(q_ref[0] + u0)

    lam_h, eq_h, xyz_true, xyz_des = [], [], [], []
    for idx in range(n):
        qr = q_ref[idx]
        u = ilc.correction_at(lambdas[idx]) if use_ilc else np.zeros(7)
        for _ in range(hold):
            plant.command_motor(qr + u)
        q_joint = plant.read_joint_encoders()      # joint-side -> ILC
        lam_h.append(lambdas[idx])
        eq_h.append(qr - q_joint)
        xyz_true.append(plant.read_tcp())           # truth -> evaluation only
        xyz_des.append(xyz_ref[idx])

    xyz_true = np.array(xyz_true); xyz_des = np.array(xyz_des)
    cart_err = np.linalg.norm(xyz_true - xyz_des, axis=1)   # meters
    return {
        "lambdas": np.array(lam_h),
        "eq": np.array(eq_h),
        "cart_err_m": cart_err,
        "rms_um": 1e6 * np.sqrt(np.mean(cart_err ** 2)),
        "max_um": 1e6 * np.max(cart_err),
        "xyz_true": xyz_true,
        "xyz_des": xyz_des,
    }


def new_ilc(n):
    """Fresh path-ILC sized to the reference, paper gains, 7 joints."""
    return PathILC(n_intervals=n, n_joints=7, Kp_ilc=1.0, Kd_ilc=0.01,
                   Nq=6, f_cutoff_hz=5.0, Ts=2e-3)
