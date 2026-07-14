# Attention Residuals Analysis

A from-scratch implementation and gradient-flow analysis of **Attention Residuals (AttnRes)** on a small decoder-only transformer, compared against **DeepCrossAttention (DCA)** and standard PreNorm residuals.

## Background

Standard residual connections in modern LLMs accumulate every preceding layer's output with fixed unit weights. The Kimi Team's *Attention Residuals* report argues this uniform aggregation causes uncontrolled hidden-state growth with depth and progressively dilutes each layer's contribution. AttnRes replaces the fixed accumulation with a softmax attention over preceding layer outputs, giving each layer learned, input-dependent weights over earlier representations.

DeepCrossAttention (Heddes et al., ICML 2025) attacks the same dilution problem from a slightly different angle, with learnable input-dependent weights and depth-wise cross-attention. It serves as the baseline here.

## The notebook

[`notebook.ipynb`](./notebook.ipynb) trains three ~19M-parameter, byte-level models on enwik8, identical in backbone, seed and batch order, differing only in how each layer's input is formed from the preceding layer outputs:

| Variant | Residual mechanism |
|---|---|
| **Baseline** | standard PreNorm residuals (fixed unit weights) |
| **DCA** | GRN-gated access to past layer outputs (Heddes et al., ICML 2025) |
| **AttnRes** | softmax attention over all past layer outputs (Kimi Team 2026), implemented from scratch |

It then checks four of the paper's claims:

1. **Loss** reproduces: all three train stably, AttnRes ahead (1.499 vs. DCA 1.522 vs. baseline 1.554 nats/byte), though at this scale "comparable loss, trains stably" is the honest reading.
2. **Output magnitudes across depth** reproduces: the baseline's grow with depth (~4 to ~17), AttnRes stays bounded. The cleanest result here, since it follows from the update rule itself.
3. **Gradient uniformity across depth** does **not** reproduce: AttnRes comes out the *least* uniform of the three, not the most. The notebook reports this as an honest negative and explains why depth 6 (vs. the paper's ~24) and Full AttnRes (vs. the paper's Block AttnRes) make the comparison unfair to begin with.
4. **Learned depth-attention patterns** reproduces: locality and embedding persistence both show up in the weight heatmap.

The notebook is self-contained. enwik8 (~35 MB) downloads itself into `data/` on first run.

## Credits

The shared GPT backbone, the DCA baseline and the training loop are adapted from Phil Wang's (lucidrains) [`deep-cross-attention`](https://github.com/lucidrains/deep-cross-attention), MIT-licensed (Copyright © 2025 Phil Wang); the enwik8 archive is fetched from the same repository. The AttnRes implementation is our own.

## References

- Kimi Team (2026). *Attention Residuals* (Technical Report). https://github.com/MoonshotAI/Attention-Residuals
- Heddes, M., Javanmard, A., Axiotis, K., Fu, G., Bateni, M. H., Mirrokni, V. (2025). *DeepCrossAttention: Supercharging Transformer Residual Connections.* ICML 2025. arXiv:2502.06785.
