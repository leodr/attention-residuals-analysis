# Attention Residuals Analysis

A from-scratch implementation and gradient-flow analysis of **Attention Residuals (AttnRes)** on a GPT-2-style transformer, with a baseline comparison against **DeepCrossAttention (DCA)**.

This repository is the deliverable for the grade bonus of the TUM seminar *Large 950879307 (S26)* (advisor: Tristan Till). The agreed scope is documented in [`proposal.md`](./proposal.md).

## Background

Standard residual connections in modern LLMs accumulate every preceding layer's output with fixed unit weights. The Kimi Team's *Attention Residuals* report argues this uniform aggregation causes uncontrolled hidden-state growth with depth and progressively dilutes each layer's contribution. AttnRes replaces the fixed accumulation with a softmax attention over preceding layer outputs, giving each layer learned, input-dependent weights over earlier representations.

DeepCrossAttention (Heddes et al., ICML 2025) attacks the same dilution problem from a slightly different angle, with learnable input-dependent weights and depth-wise cross-attention. It serves as the baseline for this project.

## Scope

The advisor-confirmed minimum scope is:

1. From-scratch implementation of Attention Residuals on top of a GPT-2-style transformer.
2. Gradient-flow analysis across layers to verify numerical stability.
3. Baseline comparison against at least one alternative — DeepCrossAttention, importable as external code.
4. Visualization of the AttnRes architecture/process.

Anything beyond this (Block AttnRes for memory efficiency, MoE integration, scaling-law experiments, mHC from DeepSeek, ResNet-style residuals) is optional extension territory.

## Deliverables

Per the seminar guidelines:

- **3–5 slides** on the AttnRes paper, structured as Intuition / Method / Discussion. Slides must call out at least one predecessor (standard residual connections / PreNorm) and any unresolved weaknesses or follow-up work.
- **A notebook** that reproduces a partial/toy result from the paper, includes one architecture/process diagram, and runs end-to-end without errors.

## Repository structure

```
attention-residuals-analysis/
├── README.md                        # This file
├── AGENTS.md                        # Guidance for AI assistants working in this repo
├── proposal.md                      # The proposal sent to the advisor
└── related_works/
    ├── Attention_Residuals.pdf      # Kimi Team — primary paper
    └── DeepCrossAttention.pdf       # Heddes et al. — baseline
```

The notebook and slides will be added as the work progresses.

## Getting started

The intended environment is Python 3.11+ with PyTorch. A typical setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib jupyter
```

Once the notebook exists, launch it with `jupyter lab` from the repo root.

## References

- Kimi Team. *Attention Residuals* (Technical Report). https://github.com/MoonshotAI/Attention-Residuals
- Heddes, M., Javanmard, A., Axiotis, K., Fu, G., Bateni, M. H., Mirrokni, V. *DeepCrossAttention: Supercharging Transformer Residual Connections.* ICML 2025. arXiv:2502.06785.
