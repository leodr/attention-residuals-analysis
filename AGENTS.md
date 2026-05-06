# AGENTS.md

Guidance for AI assistants working in this repository.

## What this repo is

A grade-bonus submission for the TUM seminar *Large 950879307 (S26)*. The deliverable is two artifacts: a 3–5 slide deck and a Jupyter notebook, both centered on the Kimi Team's *Attention Residuals* (AttnRes) paper.

The author is Leonhard Driesch. The advisor is Tristan Till. The agreed scope is captured in `proposal.md` at the repo root — read it first when joining a new session.

## Confirmed scope (do not silently expand)

The advisor explicitly *reduced* the minimum scope to keep things fair. Stick to it unless the user asks for more. The confirmed minimum is:

1. **From-scratch implementation of Attention Residuals on a GPT-2-style transformer.** "From scratch" means the AttnRes mechanism itself; the GPT-2 backbone can be a small custom implementation (preferred for didactic value) or a minimal reuse — but the AttnRes block must be hand-written, not imported.
2. **Gradient-flow analysis across layers** to verify numerical stability. Concretely: per-layer gradient norms, possibly hidden-state magnitudes across depth, comparing with-AttnRes vs. baseline.
3. **One baseline comparison.** DeepCrossAttention (DCA) is the recommended baseline and *may* be imported / use external code. Do not reimplement DCA from scratch unless asked.
4. **A visualization of the AttnRes architecture/process.** A diagram in the notebook (and reusable in the slides) is sufficient.

Out of scope unless the user explicitly opts in: Block AttnRes (memory-efficient variant), MoE integration, full scaling-law experiments, multi-head Cross (mHC) from DeepSeek, ResNet-style residual ablations, large-scale training runs.

## Slide requirements (per the seminar prompt)

- Structure: **Intuition → Method → Discussion**.
- Must clearly outline **at least one predecessor** AttnRes builds on (standard residual connections with PreNorm is the obvious one; ResNet for the broader lineage).
- Must scope out follow-up/weaknesses inside existing slides — *not* as separate slides.
- Include the same architecture/process diagram used in the notebook.

## Notebook requirements

- Must run end-to-end without errors. Treat this as a hard constraint.
- Reproduces a partial/toy result from the AttnRes paper. The proposal commits to TinyShakespeare as the toy corpus and a three-way comparison: **AttnRes vs. DeepCrossAttention vs. standard residuals**, all trained on a small GPT-2-style transformer. Targets are (a) trains stably, (b) more uniform hidden-state magnitudes and gradient norms across depth than the standard-residual baseline, (c) comparable training loss.
- For DCA, prefer lucidrains' implementation (https://github.com/lucidrains/deep-cross-attention) as the reference — the proposal explicitly anchors the AttnRes implementation style to it for a fair comparison.
- Include the architecture/process diagram inline (matplotlib, an SVG, or a Mermaid cell).
- Keep it readable: narrative markdown between code cells explaining what's happening and why.

## Key references

The two papers in `related_works/`:

- `Attention_Residuals.pdf` — Kimi Team, *Attention Residuals* (Technical Report). The primary paper.
- `DeepCrossAttention.pdf` — Heddes et al., ICML 2025. The baseline.

The Kimi reference implementation lives at https://github.com/MoonshotAI/Attention-Residuals — useful as ground truth when in doubt about a detail of the AttnRes formulation, but the in-repo implementation should remain hand-written.

For DCA, lucidrains' implementation at https://github.com/lucidrains/deep-cross-attention is the reference. The AttnRes implementation should follow the same structural conventions where reasonable, so the three-way comparison (AttnRes / DCA / standard residuals) is fair.

## Conventions

- Python 3.11+, PyTorch.
- Prefer small, readable code over clever code. This is a teaching artifact.
- Set seeds and document hardware assumptions whenever a result is reported.
- Keep dependencies minimal: torch, numpy, matplotlib, plus whatever's needed to import the DCA baseline.
- When citing the papers in markdown cells or slides, use author + year + venue (Kimi Team 2025 / Heddes et al. ICML 2025).

## File layout expectations

Expected additions as the project progresses:

- `notebook.ipynb` — the deliverable notebook at the repo root.
- `slides/` — the 3–5 slide deck (PDF + source if applicable).
- `src/` — only if the notebook gets too large; modules for the AttnRes block, the GPT-2 backbone, and the gradient-flow probes.

Do not create READMEs, requirements files, or scaffolding directories speculatively. Add them when there's content to put in them.

## When in doubt

Ask the user before extending scope, refactoring the notebook into a package, or adding new dependencies. The grading rubric rewards a clean, runnable, well-explained minimum more than an ambitious half-finished maximum.
