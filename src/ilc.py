"""
Path-parameter Iterative Learning Control with a Gaussian Q-filter.

This is a faithful re-implementation of the core algorithm from:
    M. Schwegel and A. Kugi, "A Simple Computationally Efficient Path ILC
    for Industrial Robotic Manipulators", ICRA 2024.

Key equations reproduced here (paper numbering):
    (10) PD-type ILC update, indexed by path parameter lambda (NOT time):
             alpha_i[l] = u_{i-1}[l] + Kp_ilc * e_q[l] + Kd_ilc * de_q[l]
    (14) Q-filter applied to the update:   u_i = Q * alpha_i
    (15),(16) Q is a Gaussian filter; sigma set from a -3dB cutoff freq f
    (19) discrete-time normalized convolution over a +/- Nq window

The whole point of indexing by path position l (arc-length-like, in [0,1])
rather than by time t is that the learned correction generalizes across
execution speeds: "at 30% along the path add this" is true whether you go
fast or slow. That is the paper's headline contribution.

Nothing in this file knows anything about the source of the error. It only
sees a measured joint error along the path and produces a correction. That
separation is deliberate: the ILC must not be handed the answer.
"""

import numpy as np


def gaussian_q_filter(Nq: int, f_cutoff_hz: float, Ts: float):
    """Build the discrete Gaussian Q-filter kernel (paper eqs. 15, 16, 18).

    The paper sets the standard deviation from a -3dB cutoff frequency f:
        sigma = sqrt(ln 2) / (2 * pi * f)      (eq. 16, in seconds)
    We convert that to samples using Ts, then sample a Gaussian over a
    symmetric window of (2*Nq + 1) taps and normalize it (eq. 19 divides by
    the sum of the taps, so the kernel is a weighted average -> unity DC gain).

    Returning a normalized kernel means convolution preserves the mean level
    of the correction while smoothing out non-repetitive / high-frequency
    content -- exactly what stops the ILC from learning measurement noise.
    """
    sigma_s = np.sqrt(np.log(2.0)) / (2.0 * np.pi * f_cutoff_hz)  # seconds
    sigma_samples = sigma_s / Ts                                  # in samples
    m = np.arange(-Nq, Nq + 1)
    kernel = np.exp(-(m ** 2) / (2.0 * sigma_samples ** 2))
    kernel /= kernel.sum()
    return kernel


class PathILC:
    """Path-indexed PD-type ILC with Gaussian Q-filter.

    Parameters
    ----------
    n_intervals : int
        Number of equidistant path-parameter intervals N (paper: 10000).
        The correction is stored as a length-N table u[l], l = 0..N-1.
    n_joints : int
        Dimension of the joint correction at each path index.
    Kp_ilc, Kd_ilc : float
        PD gains of the ILC update (paper Table I: 1.0 and 0.01).
    Nq : int
        Half-window of the Q-filter (paper Table I: 45).
    f_cutoff_hz : float
        -3dB cutoff of the Gaussian Q-filter (paper Table I: 5 Hz).
    Ts : float
        Sampling time used to scale the Gaussian (paper Table I: 1 ms).
    """

    def __init__(self, n_intervals, n_joints, Kp_ilc=1.0, Kd_ilc=0.01,
                 Nq=45, f_cutoff_hz=5.0, Ts=1e-3):
        self.N = int(n_intervals)
        self.n_joints = int(n_joints)
        self.Kp = Kp_ilc
        self.Kd = Kd_ilc
        self.Nq = Nq
        self.kernel = gaussian_q_filter(Nq, f_cutoff_hz, Ts)
        # Learned correction table, indexed by path parameter. Starts at zero:
        # trial 0 has no correction, which is why trial 0 shows the raw error.
        self.u = np.zeros((self.N, self.n_joints))

    # -- indexing -----------------------------------------------------------
    def index_of(self, lam):
        """Map a path parameter lambda in [0,1] to a table index l (paper:
        l = round(lambda / delta), delta = 1/N)."""
        l = int(round(lam * (self.N - 1)))
        return min(max(l, 0), self.N - 1)

    def correction_at(self, lam):
        """Return the current ILC correction for path parameter lambda.

        Because lookup is by path position, the SAME table works at any
        execution speed -- this is the speed-generalization property."""
        return self.u[self.index_of(lam)]

    # -- learning -----------------------------------------------------------
    def update_from_trial(self, lambdas, eq, deq=None):
        """Run one ILC update from a completed trial (eqs. 10 + 14/19).

        Parameters
        ----------
        lambdas : (T,) array
            Path parameter at each sample of the trial, in [0,1].
        eq : (T, n_joints) array
            Measured JOINT error e_q at each sample. In the paper this is the
            laser-tracker task error mapped to joints via the inverse Jacobian
            (eq. 12); in our MuJoCo build it is read directly from the joint
            encoders. The ILC does not care where it came from.
        deq : (T, n_joints) array, optional
            Joint velocity error. If None it is estimated by finite difference
            along the path (the paper reads it from the encoders, eq. 10).
        """
        lambdas = np.asarray(lambdas)
        eq = np.asarray(eq)
        if deq is None:
            deq = np.gradient(eq, axis=0)

        # 1) Bin the trial's samples onto the path-parameter grid, averaging
        #    samples that fall in the same interval (Algorithm 1 updates each
        #    interval once per trial; averaging is the natural batch analogue).
        alpha = self.u.copy()           # start from previous correction u_{i-1}
        acc = np.zeros((self.N, self.n_joints))
        cnt = np.zeros(self.N)
        for t in range(len(lambdas)):
            l = self.index_of(lambdas[t])
            # PD-type update term of eq. (10):
            acc[l] += self.Kp * eq[t] + self.Kd * deq[t]
            cnt[l] += 1
        seen = cnt > 0
        alpha[seen] += acc[seen] / cnt[seen, None]

        # 2) Fill path intervals that this trial never visited (e.g. partial
        #    trials / coarse sampling) by carrying the nearest known value, so
        #    the convolution below has no holes. This realizes the paper's
        #    "learning from partial trials" gracefully.
        if not seen.all():
            idx = np.where(seen)[0]
            if len(idx) > 0:
                nearest = idx[np.clip(np.searchsorted(idx, np.arange(self.N)),
                                      0, len(idx) - 1)]
                alpha[~seen] = alpha[nearest][~seen]

        # 3) Q-filter: normalized Gaussian convolution along the path index
        #    (eqs. 14, 19). 'same' keeps the table length; edge handling via
        #    reflection avoids end transients.
        u_new = np.empty_like(alpha)
        pad = self.Nq
        for j in range(self.n_joints):
            col = np.pad(alpha[:, j], pad, mode='reflect')
            # slice back to length N (pad may be 0, so avoid the [pad:-pad] trap)
            u_new[:, j] = np.convolve(col, self.kernel, mode='same')[pad:pad + self.N]
        self.u = u_new
        return self.u

    # -- convenience --------------------------------------------------------
    def correction_table(self):
        """Return a copy of the full learned correction table u[l]."""
        return self.u.copy()

    def load_table(self, table):
        """Load a correction table (e.g. one predicted by the learned layer)."""
        table = np.asarray(table)
        assert table.shape == self.u.shape
        self.u = table.copy()
