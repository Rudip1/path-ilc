# Figure index & talking points

One line per artifact: **which claim it proves** and the **key number**. All in
`outputs/`. Regenerate with `python src/run_all.py`.

Numbers are from a representative run (ML + physics ⇒ small run-to-run variation);
reproducible on a fixed machine with the pinned `requirements.txt`.

---

## Headline: KUKA iiwa14 with flexible-joint drivetrain error
*(experiments run under realistic repeatable effects — transmission error +
Stribeck friction + backlash + encoder noise)*

| File | Claim it proves | Key numbers |
|---|---|---|
| `kuka_convergence.png` | **Goal 1 — self-learn the drivetrain error from a joint-side (output) encoder**, no laser tracker. *(Stronger sensor than the paper's motor-encoder-only robot — see README "sensing assumption".)* | 34 mm → ~1.06 mm RMS, **~32×**, most in 2–3 trials |
| `kuka_speed_generalization.png` | Path-λ indexing transfers across speeds (honest: partial — velocity-dependent residual). | best at trained speed ≈ 0.99 mm |
| `kuka_path_transfer.png` | **The open problem.** A table learned on path A barely helps on path B; full relearn works but costs trials. | no-corr 42 mm → naive 14 mm → relearn 1.66 mm |
| `kuka_nn_path_transfer.png` | **Goal 2 — generalize to a new path.** a neural network predicts a correction for an **unseen** path, **zero trials**. | no-corr 35 mm → naive 5.7 mm → **NN 1.46 mm** |
| `kuka_contact_task.png` | **Paper's open problem — contact tasks.** ILC on a tool pressed on a worktable. | 18 mm → 1.06 mm, **~17×**, ~108 N press |

## Analysis figures (control-research style)

| File | Claim it proves | Key numbers |
|---|---|---|
| `trialwise_overview.png` | Faithful reproduction of the paper's Fig. 5 (error / ILC input / path speed); ILC-OFF at trial 5 returns the error, ON at trial 6 removes it. | error ±40 mm → ~1 mm |
| `encoder_validation.png` | **Answer to "is it circular?"** Joint-side output-encoder RMS and true TCP RMS fall in lock-step ⇒ minimizing what the encoder sees minimizes true accuracy ⇒ **no tracker needed to learn** (only to confirm this correlation once). Learned feedforward matches the true drivetrain error. | corr ≈ 0.97 |
| `error_along_path.png` | Where the error lives along the path, before vs after learning. | trial-0 vs converged |
| `joint_modal_properties.png` | **Mechanical view** — each joint is a torsional spring-mass-damper; natural frequencies vs the 5 Hz Q-filter cutoff; structural mode vs damped reality. | f_n ≈ 10–195 Hz |

## Realistic drivetrain effects

| File | Claim it proves | Key numbers |
|---|---|---|
| `realistic_effects_ablation.png` | ILC stays accurate under **every** realistic effect (repeatable ones learned, noise filtered). | all ≈ 1.0–1.1 mm after ILC |
| `qfilter_noise_rejection.png` | The **Gaussian Q-filter rejects encoder noise** — without it the learned correction is jagged (learning noise). | smooth vs jagged correction |
| `transmission_error_profile.png` | The injected **angle-periodic transmission error** (gear/cycloidal harmonics, "100×/rev"). | 0.3–0.8 mrad, 13–37 cyc/rev |
| `thermal_frozen_vs_online.png` | **Why an adaptive learned layer is needed.** Thermal drift defeats a frozen ILC (error creeps up as the robot warms); an online ILC tracks it. | frozen 1.0 → 2.2 mm; online flat |
| `thermal_learned_compensation.png` | **Goal 3 — a learned layer on top of ILC.** A temperature-aware model predicts the thermal-drift compensation at an **unseen** thermal state, zero trials. | frozen 4.5 mm → **learned 0.99 mm** ≈ oracle 1.05 mm |

## Animations

| File | What it shows |
|---|---|
| `kuka_tracking.gif` / `kuka_contact.gif` | KUKA tracing the path; tool-tip trace (deviation ×8) balloons off (no ILC) vs hugs it (after ILC). |
| `kuka_learning_dashboard.mp4` | 4-panel "mission control": arm + error-vs-λ collapsing + ILC inputs + convergence, animated across trials. |
| `kuka_thermal_dashboard.mp4` | Warm-up live: frozen vs online RMS diverging as joint temperature rises. |

---

## 30-second pitch
> I re-implemented the path-ILC faithfully on a real KUKA iiwa14 whose
> joints I modelled as torsional spring-mass-dampers with realistic drivetrain
> error (transmission error, friction, backlash, thermal drift, encoder noise).
> Then I went past the paper on its own stated open problems: **(1)** self-learn
> the drivetrain error from an **integrated joint-side (output) encoder** — the
> secondary encoder real high-accuracy arms carry — instead of a laser tracker
> (`encoder_validation` validates it's a
> faithful proxy for true TCP accuracy; this is a stronger sensor than the
> paper's motor-encoder-only robot, and I say so); **(2)** a learned model
> **generalizes the
> correction to an unseen path with zero trials** (`kuka_nn_path_transfer`); **(3)** a
> **temperature-aware learned layer** predicts the **thermal-drift** compensation that
> a fixed ILC table can't (`thermal_frozen_vs_online`, `thermal_learned_compensation`) — the learning-enhanced part. It's simulation,
> and I'm explicit about what that does and doesn't prove.
