# Literature Boundary

## Must-Cite / Must-Beat Work

- NAIT / Neuron-Aware Data Selection in Instruction Tuning
  (`https://arxiv.org/abs/2603.13201`,
  `https://openreview.net/forum?id=uq6UWRgzMr`). **Citation verified real
  (2026-06-19):** arXiv 2603.13201 is a genuine, checkable paper and a must-beat
  baseline; the earlier "future-dated ⇒ invalid" flag for this ID was a false
  positive and is withdrawn.
- **Quality-score selection → AlpaGasus** (Chen et al., ICLR 2024,
  `https://arxiv.org/abs/2307.08701`; repo `Lichang-Chen/AlpaGasus`). **Verified
  real (2026-06-19).** LLM-as-judge per-example quality scoring; a **must-beat**
  baseline run as a faithful reimplementation of the scoring method under the
  project license (no LLaMA-derived AlpaGasus model artifact is loaded).
- **Influence / gradient selection → LESS** (Xia et al., ICML 2024,
  `https://arxiv.org/abs/2402.04333`; repo `princeton-nlp/LESS`, **MIT**). **Verified
  real (2026-06-19).** Optimizer-aware low-rank gradient-similarity influence
  selection; a **must-beat** baseline (faithful reimplementation / MIT-licensed
  adaptation).
- **Diversity coreset → CONTEXTUAL, non-must-beat.** arXiv:2605.26004 is a **real**
  paper but it is *MAGIC* — a **multimodal / VLM** instruction coreset method, not a
  text instruction-tuning coreset anchor. It is therefore **out-of-domain** for this
  text-only LATTICE-R setting and is **dropped from must-beat** (`must_beat: false`):
  a non-must-beat secondary/contextual baseline run via our own documented text-coreset
  selector (k-center / facility-location over candidate embeddings). No "beats
  diversity_coreset" claim is licensed; a verified **text-domain** coreset citation
  would have to be pinned at run authorization before any such claim.
- Layer-selective or parameter-efficient tuning baselines.
- Retention and catastrophic-forgetting evaluation in instruction tuning.

## Boundary Statement

NeuroTrace-IT does not claim that neuron-aware selection is new. The
contribution must be that trajectory-level signatures add value over endpoint
activation features under the same data/training budget and that this value is
visible after retention and hallucination drift are counted.

## Reviewer-Risk Notes

- If endpoint-neuron selection matches the method, the core novelty fails.
- If layer policy only adds complexity, it should be an ablation, not a main
  contribution.
- If target gains come with retention collapse, the paper cannot claim reliable
  instruction tuning.

