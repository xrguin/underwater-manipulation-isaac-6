---
name: working-style
description: Project working-style protocols. Apply when planning non-trivial work, before editing code or files, when diagnosing system behaviour, or when revising slides / docs. Covers MCQ planning, hypothesis-via-data, diff plans, training-vs-deployment distribution checks, open-loop diagnostics, root-cause discipline, literature search, slide tightening, and retraction over rationalisation.
---

# Working-style protocols (this project)

Ten rules drawn from prior work on this codebase. Apply them by default; they exist because each was learned the hard way.

## 1. MCQ before planning a non-trivial action

Before any non-trivial action — planning a feature, editing more than one file, restructuring slides, retraining a model — present **4–6 lettered options** as multiple-choice questions covering the load-bearing decisions. The user answers with letters; that's the contract. No re-confirmation needed afterward; proceed and execute.

## 2. Verify hypotheses using plots and data, not narrative

Don't theorise. Run the script, generate the figure, read the actual numbers — *then* draw the conclusion. If a conclusion has been drawn without a corresponding plot or numerical inspection, treat it as a hypothesis, not a finding.

## 3. State the diff plan before editing files

When about to modify more than one file, list each change as "before / after / why" first. Wait for the go-ahead. Only then edit. This avoids the *"I changed X but it broke Y"* loop.

## 4. Open-loop test as primary diagnostic for control models

Closed-loop convergence can mask model errors via re-measurement (60 Hz feedback re-anchors bad predictions before they accumulate). Always test the model in isolation: constant input → integrate forward → compare to ground truth. If you only have closed-loop metrics, you don't know whether the model is correct or whether the controller is just inverting B and ignoring A.

## 5. Distinguish training distribution from deployment distribution

A feature / dictionary / pruning choice that is correct at deployment time can be wrong for training, and vice versa. When making a structural decision (drop a feature, restrict a sampling range), verify it holds in both distributions. *Example from this repo: pruning yaw-coupled features assuming `ψ_rel ≈ 0` was correct for deployment but wrong for training (uniform yaw); the model lost the kinematics it needed to learn.*

## 6. Check file contents and actual data before naming a mechanism

If you claim *"the regression learns wrong-sign coefficients because cos(roll) ≈ 1 in training"*, load `snapshots.npz` and verify cos(roll) actually ≈ 1 first. Don't run on plausibility; run on the data. Three iterations of *"claim → check → wrong"* on the same diagnosis is the signal that the loop has been skipped.

## 7. Use the literature when stuck

If a problem looks generic (multi-collinearity in regression, model-based RL with control, water rendering in a 3D engine), search for the standard fix before inventing one. Likely someone has already solved it (SINDy, SINDYc, Kernel EDMD, OmniSurface MDL, etc.). Cite the paper / library / API in the code or doc.

## 8. Keep slides tight (Beamer / experiment notes)

Use `\scriptsize` over wrapping; one figure = one slide; bullets longer than 3 lines means trim. Run `pdflatex -interaction=nonstopmode` and grep `Overfull` after every edit; treat any vbox > 1pt as a real overflow to fix. Drop the "what this means" bridge paragraph if the bullet list already says it.

## 9. Acknowledge "this verdict was wrong" rather than rationalise

If a follow-up test contradicts an earlier claim, **retract** the previous explanation and rebuild the diagnosis from the new evidence. Don't sell-stack the old story by attaching epicycles. Update slides, docs, and verbal claims at the same time so the record is coherent.

## 10. Don't propose a fix until you have verified the cause

Symptoms have many possible causes. Workarounds (LP filter on a noisy reference, body-local frame transform on a wrong-direction MPC) treat the symptom, not the disease, and accumulate. Confirm root cause via #2 and #6 before proposing a fix; otherwise you'll layer band-aids that have to be peeled off later.

---

## Cross-references

- `docs/experiment_notes.pdf` — diagnostic deck demonstrating these patterns on the EDMDc / Fossen work.
- `CLAUDE.md` — project conventions (what); this skill — work-style protocols (how).
