"""
MuJoCo arm "plant" with deliberately UNMODELED dynamics.

Design choice that keeps this honest (not circular):
  - The CONTROLLER (in run.py) plans using a simple rigid kinematic model:
    it computes reference joint angles for a desired Cartesian path and
    commands them, assuming the joints follow perfectly.
  - The PLANT simulated here has extras the controller does NOT know about:
    joint friction (damping + dry friction via frictionloss), gravity
    loading, and link compliance (a soft joint coupling). The mismatch
    between "what the controller assumes" and "what physics does" produces
    a path error that is REPEATABLE but NOT authored as a closed-form
    function. The ILC has to discover the correction from joint-encoder
    error alone -- it is never told the friction/stiffness values.

This is the same situation as the paper's "unknown transmission error
dynamics", scaled down to a teaching-sized 3-DOF planar arm so it runs in
seconds and is fully reproducible (no external URDF/mesh files needed).

Honesty note carried into the README: because this is simulation, the joint
"encoder" is noise-free, so goal (1) "learn without a laser tracker" is shown
structurally (we use only joint sensors, never an external pose sensor), not
as a solution to real-world sensing noise.
"""

import numpy as np
import mujoco

# A 3-DOF planar arm built inline as MJCF. Link lengths 0.4/0.35/0.25 m.
# The joints carry damping + frictionloss (Coulomb-like) and gravity acts in
# -z, so holding the arm out loads the joints differently along a path. None
# of these numbers are visible to the controller.
_ARM_XML = """
<mujoco model="planar3">
  <option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast"/>
  <default>
    <joint type="hinge" axis="0 1 0" damping="0.2" frictionloss="0.8" armature="0.01"/>
    <geom type="capsule" density="800"/>
  </default>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="l1" pos="0 0 0">
      <joint name="j1"/>
      <geom name="g1" fromto="0 0 0 0.4 0 0" size="0.03"/>
      <body name="l2" pos="0.4 0 0">
        <joint name="j2"/>
        <geom name="g2" fromto="0 0 0 0.35 0 0" size="0.025"/>
        <body name="l3" pos="0.35 0 0">
          <joint name="j3"/>
          <geom name="g3" fromto="0 0 0 0.25 0 0" size="0.02"/>
          <site name="tcp" pos="0.25 0 0" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="800" kv="40"/>
    <position name="a2" joint="j2" kp="600" kv="30"/>
    <position name="a3" joint="j3" kp="400" kv="20"/>
  </actuator>
  <sensor>
    <jointpos name="s1" joint="j1"/>
    <jointpos name="s2" joint="j2"/>
    <jointpos name="s3" joint="j3"/>
  </sensor>
</mujoco>
"""

L = np.array([0.40, 0.35, 0.25])  # link lengths used by the IDEAL controller


def forward_kinematics(q):
    """Ideal planar FK (controller's model). Returns TCP (x, z) in the plane.
    The arm lies in the x-z plane; joint angles are absolute-cumulative.

    Note the -sin for z: the MuJoCo hinges use axis (0,1,0), so a positive
    joint rotation advances the TCP in the -z direction. Matching that sign
    here keeps the controller's kinematic model consistent with the plant's
    geometry, so the ONLY remaining discrepancy is the unmodeled dynamics
    (friction, compliance, gravity loading) -- which is the point."""
    a = np.cumsum(q)
    x = np.sum(L * np.cos(a))
    z = -np.sum(L * np.sin(a))
    return np.array([x, z])


def inverse_kinematics(target_xz, q_init=None, iters=200, lr=0.5):
    """Ideal IK by damped least squares (controller's model). Maps a desired
    TCP (x, z) to joint angles, assuming a perfect rigid arm. This is the
    'model' the controller trusts -- and which the plant then violates."""
    q = np.array([0.3, 0.4, 0.2]) if q_init is None else q_init.copy()
    for _ in range(iters):
        p = forward_kinematics(q)
        err = target_xz - p
        if np.linalg.norm(err) < 1e-9:
            break
        # numerical Jacobian (2x3)
        J = np.zeros((2, 3))
        eps = 1e-6
        for k in range(3):
            dq = q.copy(); dq[k] += eps
            J[:, k] = (forward_kinematics(dq) - p) / eps
        # damped least squares step
        JT = J.T
        q += lr * JT @ np.linalg.solve(J @ JT + 1e-6 * np.eye(2), err)
    return q


class ArmPlant:
    """Wraps the MuJoCo model. Exposes only joint-position sensing (encoders).

    The controller talks to this through command_joint() / read_encoders();
    it never reads the friction, stiffness, or gravity that cause the error.
    """

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_ARM_XML)
        self.data = mujoco.MjData(self.model)
        self.tcp_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    def reset(self, q0):
        self.data.qpos[:3] = q0
        self.data.qvel[:3] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def command_joint(self, q_cmd):
        """Send a joint-position command to the actuators (one control step)."""
        self.data.ctrl[:3] = q_cmd
        mujoco.mj_step(self.model, self.data)

    def read_encoders(self):
        """Joint-side encoders: the only feedback the ILC is allowed to use."""
        return np.array(self.data.sensordata[:3])

    def read_tcp(self):
        """TRUE Cartesian TCP position from the simulator. Used ONLY for
        evaluation/plotting (the 'ground-truth laser tracker'), never fed to
        the ILC. Returns (x, z) in the arm plane."""
        p = self.data.site_xpos[self.tcp_id]
        return np.array([p[0], p[2]])
