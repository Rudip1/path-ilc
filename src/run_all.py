"""
Regenerate every result in one command.

    python src/run_all.py            # everything (figures + GIFs + MP4s)
    python src/run_all.py --quick    # figures/experiments only (skip slow videos)

Each script is run in its own process with MUJOCO_GL=glx (the working render
backend on this machine). Outputs land in ../outputs.
"""

import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (script, description). "core" = experiments + figures; "media" = GIFs / MP4s.
CORE = [
    ("experiments_toy.py", "toy 3-DOF demo (convergence, transfer, neural net)"),
    ("experiments_kuka.py", "KUKA headline experiments (convergence, speed, transfer, neural net)"),
    ("experiments_contact.py", "KUKA contact task"),
    ("figures_analysis.py", "analysis figures (trialwise overview, encoder validation, modal)"),
    ("figures_realism.py", "realistic-effects figures (ablation, Q-filter, thermal)"),
]
MEDIA = [
    ("render_toy.py", "toy arm GIF"),
    ("render_kuka.py", "KUKA tracking GIF"),
    ("render_contact.py", "KUKA contact GIF"),
    ("dashboard_learning.py", "learning dashboard MP4"),
    ("dashboard_thermal.py", "thermal-adaptation dashboard MP4"),
]


def run(script, desc):
    print(f"\n=== {script}  —  {desc} ===", flush=True)
    t0 = time.time()
    env = dict(os.environ, MUJOCO_GL="glx")
    r = subprocess.run([PY, os.path.join(HERE, script)], env=env)
    dt = time.time() - t0
    status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
    print(f"--- {script}: {status} in {dt:.0f}s", flush=True)
    return r.returncode == 0


def main():
    quick = "--quick" in sys.argv
    scripts = CORE + ([] if quick else MEDIA)
    print(f"Running {len(scripts)} scripts" + (" (quick: no videos)" if quick else ""))
    t0 = time.time()
    ok = sum(run(s, d) for s, d in scripts)
    print(f"\nDone: {ok}/{len(scripts)} succeeded in {time.time()-t0:.0f}s. "
          f"See outputs/.")


if __name__ == "__main__":
    main()
