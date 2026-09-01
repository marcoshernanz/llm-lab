# TFG feasibility probes

These are bounded diagnostics for selecting a 2026–27 thesis question. They are not language-model-scale evidence and should not be presented as confirming a DeepSeek-V4 mechanism or a deployable quantization method.

The exact current result table and limitations are in [`RESULTS.md`](./RESULTS.md).

## Historical routing

`anticipatory_routing_probe.py` builds a tiny autoregressive Transformer with three top-2 MoE blocks. It trains one pre-shock trajectory per seed, then forks model and optimizer state into synchronous and historical-route recovery branches. Both branches perform the same no-gradient future forward so additional route-precomputation work is matched.

The shock persistently attenuates one first-block expert. Historical routing replays full route indices computed `delay` updates earlier while current features and gradients remain active.

Run the three-seed default:

```bash
uv run python experiments/tfg/anticipatory_routing_probe.py \
  --output experiments/tfg/artifacts/anticipatory-routing-probe.json
```

Severity probes use `--shock-scale 3` or `--shock-scale 8`. The paired effects are reported as historical minus synchronous, so a negative value favors historical routing.

Limitations:

- synthetic Markov-domain data;
- tiny model and short training trajectory;
- injected permanent parameter shock;
- training-batch rather than fixed-panel validation metric;
- full route replay only, without decomposition of stale weights and representations.

The thesis plan explicitly addresses these limitations before any confirmatory claim.

## Recurrent-state quantization

`recurrent_state_quantization_probe.py` updates a synthetic Gated-Delta-style recurrent matrix and quantizes the persistent state after every write. It compares per-head symmetric INT4 and INT8 storage over several decay constants and sequence lengths.

```bash
uv run python experiments/tfg/recurrent_state_quantization_probe.py \
  --output experiments/tfg/artifacts/recurrent-state-quantization-probe.json
```

This tests numerical error accumulation only. It does not test a language model, calibrated mixed precision, a packed representation, or kernel speed.

## Verification

```bash
uv run ruff check experiments/tfg tests/test_tfg_probes.py
uv run pytest -q tests/test_tfg_probes.py
```

The regression tests verify branch-order invariance after optimizer checkpoint loading and the expected INT8/INT4 error ordering.
