# Proposal

Sent to Tristan Till on 2026-05-06, in reply to his message of 2026-04-30 confirming the topic and reduced scope.

---

Ok, here's my revised proposal:

**Topic:** Attention Residuals (Kimi Team, 2025)

**Slides:** First I'll start with the predecessor (just res connections), go into the problems (hidden state grows unbounded and contribution of layers is diluted when tackling the growth through normalization). Then on the method slide, I go into the AttnRes, so how the fixed sum is replaced with softmax attention over earlier layer outputs. I'd put an architecture diagram here, kind of similar to Figure 1 in the paper, but how I find it most intuitive to understand. Then for the discussion slide, I'd go into how Block AttnRes improves the performance and the impact on training. I'll also mention that the method is fairly new and there's not much work building on it. Should I also compare the method to DeepMind paper here or just in the notebook?

**Notebook:** From-scratch AttnRes on GPT-2 style model (I'd try to keep it similar to [this implementation](https://github.com/lucidrains/deep-cross-attention) of DeepCrossAttention for fair comparison), trained on toy corpus (TinyShakespeare would be my choice), aiming to reproduce/show the claims from the paper about hidden-state magnitudes and gradients. Compare this across AttnRes, DeepCrossAttention, Standard Residuals.

Does this work?

Best, Leo
