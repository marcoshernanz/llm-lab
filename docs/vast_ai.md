# Vast.ai Notes

Short repo-local note for searching GPUs on Vast.ai.

## Search

Default `5090` search:

```bash
vastai search offers \
  'gpu_name=RTX_5090 num_gpus=1 reliability>0.99 dlperf>195 pcie_bw>20 dph<0.45' \
  --storage 60 \
  --limit 20 \
  -o 'dph,reliability-,pcie_bw-,dlperf-' \
  --raw
```

Region-aware variant:

```bash
vastai search offers \
  'gpu_name=RTX_5090 num_gpus=1 reliability>0.995 dlperf>195 pcie_bw>20 dph<0.45 geolocation notin [VN,KR]' \
  --storage 60 \
  --limit 20 \
  -o 'dph,reliability-,pcie_bw-,dlperf-' \
  --raw
```

## What To Look For

- low `dph`
- high `reliability`
- high `pcie_bw`
- solid `dlperf`
- reasonable `storage_cost`
- reasonable bandwidth cost

For single-GPU runs, prefer:

- `verified=true`
- `num_gpus=1`
- `reliability > 0.99`
- `pcie_bw > 20`
- `dlperf > 195`

Use geography as a tiebreaker, not the main filter.
