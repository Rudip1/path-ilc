"""
Flexible-joint KUKA iiwa14 plant, built from the MuJoCo Menagerie model.

Why this is the faithful version of the problem
-----------------------------------------------
High-accuracy industrial robotics targets sub-10 um absolute accuracy without
external measurement systems, using integrated joint-side encoders, by
self-learning the drivetrain errors of geared (e.g. cycloidal) drive modules.
The dominant such error is JOINT ELASTICITY / TRANSMISSION ERROR: the motor
turns by theta_m, but the
link actually arrives at a slightly different angle theta_l because the gear
train is not perfectly stiff and deflects under load (gravity, acceleration).

We reproduce that here by turning every axis of the real KUKA iiwa14 into a
two-mass flexible joint:

      motor DOF  --[ torsional spring K + damper d ]--  link DOF (carries the
      (actuated)                                          real link + payload)

  * The CONTROLLER (kuka_simulation.py) plans on the RIGID iiwa14 (load_rigid_model):
    it does inverse kinematics assuming theta_link == theta_motor_command.
  * The PLANT (build_flexible_model) has the springs, so under gravity/motion
    the link lags the motor by an unknown, pose- and load-dependent amount.
  * The motor-side encoder reads theta_m (what the controller assumes); the
    JOINT-SIDE / OUTPUT encoder reads the true link angle
    theta_l = theta_m + deflection (read_joint_encoders). The ILC learns the
    drivetrain error from that joint-side encoder; the true Cartesian TCP
    (read_tcp) is read for EVALUATION ONLY, never fed to the ILC.

IMPORTANT sensing caveat (see README "The sensing assumption"):
  The paper needs a laser tracker precisely because MOTOR-side encoders cannot
  observe the link dynamics. Here we instead assume a JOINT-SIDE / SECONDARY /
  OUTPUT encoder that *directly* sees the true link angle -- an integrated
  joint-side encoder, a real sensor on high-accuracy robots, but a STRONGER
  sensing assumption than the paper's
  motor-encoder-only robot. That is what makes learning-without-a-tracker
  possible here; it is not the paper's harder motor-encoder-only problem.

Everything (meshes, link masses, inertias, kinematics) comes from the real
Menagerie model; we only add the drivetrain compliance + friction that a real
robot has and a rigid model ignores.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
import numpy as np
import mujoco

from robot_descriptions import iiwa14_mj_description

_PKG = iiwa14_mj_description.PACKAGE_PATH
_MJCF = iiwa14_mj_description.MJCF_PATH
_ASSETS = os.path.join(_PKG, "assets")

JOINTS = [f"joint{i}" for i in range(1, 8)]
HOME = np.array([0.0, 0.6, 0.0, -1.4, 0.0, 1.2, 0.0])  # a well-conditioned pose

# Per-axis drivetrain parameters (root joints stiffer/heavier, wrist softer).
# Tuned so gravity/motion produce a prominent, realistic mm-level raw TCP
# error that the ILC can then drive down -- analogous to the paper's setup.
STIFFNESS = np.array([18000., 16000., 12000., 9000., 6000., 4000., 3000.])  # Nm/rad
DAMPING_L = np.array([50., 50., 35., 28., 18., 12., 8.])                     # Nms/rad
FRICTION_L = np.array([2.0, 2.0, 1.3, 1.0, 0.6, 0.4, 0.25])                  # Nm (Coulomb)
ARMATURE_M = np.array([1.2, 1.2, 0.6, 0.5, 0.2, 0.15, 0.1])                  # reflected rotor inertia


@dataclass
class Effects:
    """Toggles + parameters for the realistic drivetrain effects (see the
    'deep research' roadmap). All OFF by default, so the basic experiments are
    unchanged; `Effects.realistic()` turns everything on.

    Physics (per joint i, on the LINK degree of freedom):
      * transmission error: an angle-periodic torque K_i*eps_i(theta) makes the
        link settle at a pose-periodic offset -- a true gear/cycloidal TE.
      * Stribeck: extra low-speed friction (Fs-Fc)e^{-(v/vs)^2} -> stick-slip.
      * backlash: the elastic stiffness becomes a dead-zone spring (gear lost
        motion); the native MuJoCo stiffness is set to 0 and applied here.
      * thermal: a per-joint temperature heated by friction power adds a slow,
        drifting offset; the state PERSISTS across trials (non-repetitive).
      * encoder_noise: Gaussian noise + finite-resolution quantization on the
        joint-side encoder reading.
    """
    te: bool = False
    stribeck: bool = False
    backlash: bool = False
    thermal: bool = False
    encoder_noise: bool = False
    # transmission-error harmonics
    te_amp: np.ndarray = field(default_factory=lambda: 1e-3 * np.array(
        [0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.3]))                    # rad
    te_n1: np.ndarray = field(default_factory=lambda: np.array(
        [17, 19, 23, 13, 29, 31, 37], float))                    # cycles/rev
    te_phase: np.ndarray = field(default_factory=lambda: np.array(
        [0.3, 1.1, 2.0, 0.7, 2.6, 1.7, 0.4]))
    # Stribeck friction
    strib_ratio: float = 1.7        # Fs / Fc
    strib_vs: float = 0.04          # Stribeck velocity [rad/s]
    # backlash dead-zone (half-width) -- ~arc-min lost motion
    backlash_b: np.ndarray = field(default_factory=lambda: 1e-3 * np.array(
        [0.5, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]))                    # rad
    # thermal model (small mass + slow cooling so heat accumulates over a
    # multi-trial session -> a slow, non-repetitive warm-up drift)
    therm_C: float = 18.0           # heat capacity (heating rate scale)
    therm_tau: float = 350.0        # cooling time constant [s]
    therm_k: np.ndarray = field(default_factory=lambda: 1e-3 * np.array(
        [6, 6, 5, 5, 4, 3, 3]))                                  # rad offset per unit temp
    # encoder
    enc_bits: int = 19              # ~12 urad resolution over +/- pi
    enc_sigma: float = 2.0e-5       # rad RMS noise

    @classmethod
    def realistic(cls, **over):
        return cls(te=True, stribeck=True, backlash=True, thermal=True,
                   encoder_noise=True, **over)

    @classmethod
    def realistic_repeatable(cls, **over):
        """All realistic effects EXCEPT thermal drift -- i.e. the repeatable /
        filterable ones (transmission error, friction, backlash, encoder noise).
        Used for the headline convergence/transfer/learned-layer experiments, where the
        non-repetitive thermal drift is studied separately."""
        return cls(te=True, stribeck=True, backlash=True, thermal=False,
                   encoder_noise=True, **over)

    @property
    def any_joint_force(self):
        return self.te or self.stribeck or self.backlash or self.thermal


def _find_parent(root, child):
    for el in root.iter():
        if child in list(el):
            return el
    return None


def build_flexible_model(stiffness=STIFFNESS, damping=DAMPING_L,
                         friction=FRICTION_L, armature=ARMATURE_M,
                         marker_xyz=None, contact_z=None,
                         contact_center=(0.55, 0.0), effects=None):
    """Return an MjModel of the iiwa14 with flexible (elastic) joints.

    Each original hinge `jointi` becomes a passive elastic `jointi_link`
    (torsional spring `stiffness[i]`, damper, Coulomb friction) inside the real
    link body, and a new actuated `jointi_motor` is inserted on a massless
    rotor body just outboard of it. Actuators drive the motor joints.
    """
    tree = ET.parse(_MJCF)
    root = tree.getroot()

    # 1) load meshes by absolute path regardless of CWD
    comp = root.find("compiler")
    comp.set("meshdir", _ASSETS)

    # large offscreen framebuffer + brighter lighting for clean renders
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "960"})
    ET.SubElement(vis, "headlight", {"ambient": "0.4 0.4 0.4",
                                     "diffuse": "0.6 0.6 0.6"})

    # 2) drop the home keyframe: nq changes (7 -> 14) so its qpos won't fit
    for kf in root.findall("keyframe"):
        root.remove(kf)

    # 3) split every joint into motor (actuated) + link (elastic)
    for i, jname in enumerate(JOINTS):
        # the joint element and the body that carries it
        joint_el = link_body = None
        for body in root.iter("body"):
            j = body.find(f"joint[@name='{jname}']")
            if j is not None:
                joint_el, link_body = j, body
                break
        gp = _find_parent(root, link_body)

        # rotor body inherits the link's placement; link then sits at identity
        rotor = ET.Element("body", {"name": f"rotor{i+1}"})
        rotor.set("pos", link_body.get("pos", "0 0 0"))
        if link_body.get("quat") is not None:
            rotor.set("quat", link_body.get("quat"))
        link_body.set("pos", "0 0 0")
        if "quat" in link_body.attrib:
            del link_body.attrib["quat"]
        # rotor needs a (tiny) inertia to be a valid moving body; the real
        # rotational inertia of this DOF comes from `armature` on its joint.
        ET.SubElement(rotor, "inertial",
                      {"mass": "0.01", "pos": "0 0 0",
                       "diaginertia": "1e-4 1e-4 1e-4"})
        motor = ET.SubElement(rotor, "joint", {
            "name": f"{jname}_motor",
            "class": joint_el.get("class", "joint1"),
            "axis": "0 0 1",
            "armature": f"{armature[i]:.5f}",
            "damping": "1.0",
        })

        # convert the original joint in-place into the elastic link joint
        joint_el.set("name", f"{jname}_link")
        if "class" in joint_el.attrib:
            del joint_el.attrib["class"]
        joint_el.set("axis", "0 0 1")
        joint_el.set("limited", "false")
        # with backlash, the elastic stiffness is a dead-zone spring applied in
        # software (see FlexArmPlant._joint_disturbance), so zero it here.
        k_native = 0.0 if (effects and effects.backlash) else stiffness[i]
        joint_el.set("stiffness", f"{k_native:.1f}")
        joint_el.set("damping", f"{damping[i]:.3f}")
        joint_el.set("frictionloss", f"{friction[i]:.3f}")
        joint_el.set("armature", "0.001")

        # reparent: gp -> rotor -> link_body
        gp.remove(link_body)
        gp.append(rotor)
        rotor.append(link_body)

    # 4) retarget actuators from jointi to jointi_motor
    for act in root.find("actuator"):
        act.set("joint", f"{act.get('joint')}_motor")

    # 5) sensors: motor-side + joint-side encoders, and the true TCP position
    sensor = ET.SubElement(root, "sensor")
    for jname in JOINTS:
        ET.SubElement(sensor, "jointpos",
                      {"name": f"{jname}_m", "joint": f"{jname}_motor"})
    for jname in JOINTS:
        ET.SubElement(sensor, "jointpos",
                      {"name": f"{jname}_l", "joint": f"{jname}_link"})
    ET.SubElement(sensor, "framepos",
                  {"name": "tcp", "objtype": "site", "objname": "attachment_site"})

    # 5b) optional CONTACT task: a table whose top is at z=contact_z, plus a
    # tool-tip collision sphere on link7. They use a private contact bitmask
    # (contype/conaffinity = 2) so ONLY the tool touches the table -- the rest
    # of the arm and its self-collisions are unaffected. Gravity makes the
    # (flexible) arm press the tool onto the table; sliding then induces a
    # repeatable friction + elasticity disturbance for the ILC to learn.
    if contact_z is not None:
        wb = root.find("worldbody")
        cx, cy = contact_center
        half_h = 0.25
        # disable ALL existing collisions, so the only contact in the scene is
        # tool<->table (no spurious arm self-collisions when reaching down).
        for g in root.iter("geom"):
            g.set("contype", "0")
            g.set("conaffinity", "0")
        ET.SubElement(wb, "geom", {
            "name": "worktable", "type": "box",
            "pos": f"{cx:.4f} {cy:.4f} {contact_z - half_h:.4f}",
            "size": f"0.35 0.35 {half_h:.4f}",
            "contype": "2", "conaffinity": "2",
            "friction": "1.0 0.01 0.001",
            "rgba": "0.55 0.45 0.35 1"})
        # tool tip on link7 (at the attachment site location)
        for body in root.iter("body"):
            if body.get("name") == "link7":
                ET.SubElement(body, "geom", {
                    "name": "tooltip", "type": "sphere", "size": "0.012",
                    "pos": "0 0 0.045",
                    "contype": "2", "conaffinity": "2",
                    "friction": "1.0 0.01 0.001",
                    "rgba": "0.1 0.8 0.2 1"})
                break

    # 6) optional: draw the desired path as static spheres (visual only) so the
    # rendered/live view shows what the TCP should follow.
    if marker_xyz is not None:
        wb = root.find("worldbody")
        for p in marker_xyz:
            ET.SubElement(wb, "geom", {
                "type": "sphere", "size": "0.008",
                "pos": f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}",
                "rgba": "1.0 0.85 0.1 0.95",
                "contype": "0", "conaffinity": "0"})

    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)


def load_rigid_model():
    """The ideal rigid iiwa14 the CONTROLLER plans on (no compliance)."""
    return mujoco.MjModel.from_xml_path(_MJCF)


# --------------------------------------------------------------------------
# Controller-side kinematics on the RIGID model (the model the controller
# trusts). FK by setting qpos + mj_forward; IK by damped least squares on the
# site Jacobian. These never see the flexible plant's springs.
# --------------------------------------------------------------------------
class RigidKinematics:
    def __init__(self):
        self.model = load_rigid_model()
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    def fk(self, q):
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site].copy()

    def ik(self, target_xyz, q_init=None, iters=300, tol=1e-6):
        q = HOME.copy() if q_init is None else q_init.copy()
        jacp = np.zeros((3, self.model.nv))
        for _ in range(iters):
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            p = self.data.site_xpos[self.site]
            err = target_xyz - p
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site)
            J = jacp[:, :7]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)
            q = np.clip(q + dq, self.model.jnt_range[:7, 0],
                        self.model.jnt_range[:7, 1])
        return q


class FlexArmPlant:
    """The flexible KUKA. Talks to the controller via motor commands and
    exposes motor-side + joint-side encoders. True TCP for evaluation only."""

    def __init__(self, effects=None, seed=0, **params):
        self.fx = effects or Effects()
        self.model = build_flexible_model(effects=self.fx, **params)
        self.data = mujoco.MjData(self.model)
        # qpos addresses of motor and link joints, in axis order
        self.qadr_m = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_motor")]
            for j in JOINTS])
        self.qadr_l = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_link")]
            for j in JOINTS])
        self.dofadr_l = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_link")]
            for j in JOINTS])
        self.dofadr_m = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_motor")]
            for j in JOINTS])
        self.tcp_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.temp = np.zeros(7)          # joint temperatures (persist across trials)
        self.thermal_frozen = False      # if True, temperature is held fixed
        self.rng = np.random.default_rng(seed)

    def set_temperature(self, temp_vec):
        """Set the joint temperature state and hold it fixed (for studying a
        specific thermal operating point, e.g. the temperature-aware learned layer)."""
        self.temp[:] = np.asarray(temp_vec, float)
        self.thermal_frozen = True

    def freeze_thermal(self, flag=True):
        self.thermal_frozen = flag

    def reset(self, q_motor, reset_thermal=False):
        # Each link joint's coordinate is the elastic deflection RELATIVE to its
        # rotor, so its rest value is 0 (link aligned with motor). The absolute
        # joint angle lives in the motor coordinate. Thermal state PERSISTS
        # across trials (it is the non-repetitive part) unless asked otherwise.
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.qadr_m] = q_motor
        self.data.qpos[self.qadr_l] = 0.0
        self.data.ctrl[:7] = q_motor
        self.data.qfrc_applied[:] = 0.0
        if reset_thermal:
            self.temp[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _joint_disturbance(self):
        """Disturbance torque on each LINK dof from the enabled effects."""
        fx = self.fx
        theta = self.data.qpos[self.qadr_m] + self.data.qpos[self.qadr_l]  # abs angle
        defl = self.data.qpos[self.qadr_l]                                 # spring coord
        v_abs = self.data.qvel[self.dofadr_m] + self.data.qvel[self.dofadr_l]  # joint speed
        tau = np.zeros(7)
        if fx.backlash:                  # dead-zone spring replaces native stiffness
            dz = np.sign(defl) * np.maximum(np.abs(defl) - fx.backlash_b, 0.0)
            tau += -STIFFNESS * dz
        if fx.te:                        # angle-periodic transmission error
            eps = fx.te_amp * np.sin(fx.te_n1 * theta + fx.te_phase)
            tau += STIFFNESS * eps
        if fx.stribeck:                  # extra low-speed (Stribeck) friction
            tau += -(fx.strib_ratio - 1.0) * FRICTION_L * np.exp(
                -(v_abs / fx.strib_vs) ** 2) * np.tanh(v_abs / 1e-3)
        if fx.thermal:                   # slow thermal-expansion offset
            tau += STIFFNESS * (fx.therm_k * self.temp)
        return tau

    def _update_thermal(self):
        """Heat each joint by its friction power; cool toward ambient."""
        if not self.fx.thermal or self.thermal_frozen:
            return
        dt = self.model.opt.timestep
        v_abs = self.data.qvel[self.dofadr_m] + self.data.qvel[self.dofadr_l]
        power = FRICTION_L * np.abs(v_abs)          # frictional heating (gross motion)
        self.temp += dt * (power / self.fx.therm_C - self.temp / self.fx.therm_tau)

    def command_motor(self, q_cmd):
        self.data.ctrl[:7] = q_cmd
        if self.fx.any_joint_force:
            self.data.qfrc_applied[self.dofadr_l] = self._joint_disturbance()
        mujoco.mj_step(self.model, self.data)
        self._update_thermal()

    def read_motor_encoders(self):
        """Motor-side encoders: the angle the controller assumes the link is at."""
        return self.data.qpos[self.qadr_m].copy()

    def read_joint_encoders(self):
        """Joint-side encoders: the TRUE absolute link angle = motor + elastic
        deflection. This is the only feedback the ILC may use (eq. 8/12).
        With encoder_noise on, add Gaussian noise + finite-resolution quantization."""
        q = self.data.qpos[self.qadr_m] + self.data.qpos[self.qadr_l]
        if self.fx.encoder_noise:
            q = q + self.rng.normal(0.0, self.fx.enc_sigma, size=7)
            res = (2 * np.pi) / (2 ** self.fx.enc_bits)
            q = np.round(q / res) * res
        return q.copy()

    def read_deflection(self):
        """The elastic transmission error per axis (joint-side - motor-side)."""
        return self.data.qpos[self.qadr_l].copy()

    def read_temperature(self):
        """Per-joint temperature state (above ambient). Diagnostics / learned-model input."""
        return self.temp.copy()

    def read_tcp(self):
        """TRUE Cartesian TCP (the 'laser tracker'): evaluation only."""
        return self.data.site_xpos[self.tcp_site].copy()

    def read_contact_force(self):
        """Total contact-force magnitude on the tool tip (0 if not touching).
        For the contact task; evaluation/diagnostics only."""
        total = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            total += np.linalg.norm(f6[:3])
        return total
