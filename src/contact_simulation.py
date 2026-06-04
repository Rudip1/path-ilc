"""
CONTACT task for the flexible KUKA -- the paper's stated future-work item
("applications with contacts between the manipulator and the environment").

A polishing/deburring-style task: the tool tip is dragged along a horizontal
circle while resting on a worktable. Gravity presses the (flexible) arm onto
the surface (~90 N), so as the tool slides, sliding FRICTION plus the joint
elasticity produce an extra, path-direction-dependent, REPEATABLE in-plane
tracking error -- exactly the kind of repeatable contact disturbance an ILC
can cancel. The same path-parameter PD-ILC (ilc.py), unchanged, learns it from
joint-side encoders only.

The surface defines the tool height, so the meaningful accuracy metric is the
IN-PLANE path error; we report full 3D Cartesian RMS (z stays sub-mm).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from kuka_plant import FlexArmPlant, RigidKinematics, HOME  # noqa: E402
from kuka_simulation import run_trial, new_ilc  # noqa: E402

_KIN = RigidKinematics()

TOOL_R = 0.012             # tool-tip sphere radius
PATH_Z = 0.32             # tool-center height of the contact path
TABLE_Z = PATH_Z - TOOL_R  # worktable top
CENTER = (0.55, 0.0)
RADIUS = 0.10


def contact_point(lam):
    """Desired tool-center point on the surface circle, at path param lam."""
    a = 2 * np.pi * lam
    return np.array([CENTER[0] + RADIUS * np.cos(a),
                     CENTER[1] + RADIUS * np.sin(a),
                     PATH_Z])


def make_contact_reference(n=200):
    lambdas = np.linspace(0.0, 1.0, n)
    xyz = np.array([contact_point(l) for l in lambdas])
    q = np.zeros((n, 7)); q_prev = _KIN.ik(xyz[0], q_init=HOME)
    for i, p in enumerate(xyz):
        q_prev = _KIN.ik(p, q_init=q_prev); q[i] = q_prev
    return lambdas, xyz, q


def make_contact_plant(marker_xyz=None, effects=None):
    return FlexArmPlant(contact_z=TABLE_Z, contact_center=CENTER,
                        marker_xyz=marker_xyz, effects=effects)


if __name__ == "__main__":
    import os as _os
    _os.environ.setdefault("MUJOCO_GL", "glx")
    plant = make_contact_plant()
    lambdas, xyz, qref = make_contact_reference(n=200)
    ilc = new_ilc(len(lambdas))
    print("contact-task ILC convergence (tool dragged on a worktable):")
    for i in range(8):
        r = run_trial(plant, ilc, lambdas, xyz, qref,
                      hold=30, settle=500, use_ilc=(i > 0))
        F = plant.read_contact_force()
        print("  trial %d  RMS=%7.1f um  max=%7.1f um  (press ~%.0f N)"
              % (i, r["rms_um"], r["max_um"], F))
        ilc.update_from_trial(r["lambdas"], r["eq"])
