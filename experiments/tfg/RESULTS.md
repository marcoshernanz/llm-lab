# Feasibility-probe results

Updated: 2026-08-31

These results select and constrain a thesis design. They are not evidence that Anticipatory Routing improves real language-model training.

## Historical-route probe

All rows use three paired seeds. Values are historical minus synchronous; negative values favor history. The model has three MoE blocks, four experts per block, top-2 routing, width 64, and a synthetic multi-domain autoregressive task. At step 140, the output weights of the busiest first-block expert are multiplied by the shock scale. Historical route indices are used for 64 recovery steps.

| Shock scale | History delay | Mean peak-loss delta | Mean positive-excess-loss AUC delta | Mean recovery-step delta | Historical route disagreement |
|---:|---:|---:|---:|---:|---:|
| 3× | 8 | +0.0017 | +0.3478 | -1.33 | 0.126–0.176 |
| 5× | 8 | -0.1129 | +0.0065 | -1.33 | 0.146–0.187 |
| 8× | 8 | -0.1734 | +0.1921 | +13.33 | 0.159–0.218 |
| 5× | 2 | -0.1064 | +1.0027 | +10.33 | 0.076–0.090 |

The most stable pattern is metric disagreement: historical routes often lower the single worst loss, but they do not reliably lower cumulative damage and can delay recovery substantially. The two-step delay has less route disagreement than the eight-step delay yet worse cumulative damage at 5×, so neither “less staleness” nor “more route stability” is a sufficient explanation.

Critical limitations:

- the baseline is the preceding training-loss mean, not a paired no-shock branch;
- the measured loss is on changing training batches, not a fixed validation panel;
- the perturbation is a persistent weight amplification, not a natural loss spike;
- only full cached routes are tested; historical router weights and frozen-router controls are absent;
- the task/model are synthetic and tiny;
- three seeds are insufficient for a confirmatory uncertainty claim.

The next experiment must add no-shock synchronous/historical branches and fixed-panel validation before treating these differences as causal damage.

Artifacts:

- `artifacts/anticipatory-routing-probe-shock3.json`
- `artifacts/anticipatory-routing-probe.json`
- `artifacts/anticipatory-routing-probe-shock8.json`
- `artifacts/anticipatory-routing-probe-delay2.json`

## Recurrent-state quantization probe

The probe updates a synthetic Gated-Delta-style recurrent matrix and symmetrically quantizes the persistent state after each write.

Trajectory relative-L2 at 2,048 writes:

| Decay | INT4 | INT8 |
|---:|---:|---:|
| 0.99 | 0.755 | 0.039 |
| 0.999 | 0.991 | 0.050 |

This verifies error accumulation under repeated state writes. It does not evaluate language-model quality, calibrated mixed precision, packed memory, or kernel performance. DAMP and adjacent literature—not this diagnostic—are the reason the idea is rejected as a thesis.

Artifact: `artifacts/recurrent-state-quantization-probe.json`.
