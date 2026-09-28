# FlyWire/Codex v783 annotation tables — provenance

All files in this directory were downloaded on **2026-09-28** from the public
FlyWire Codex data release for exactly the v783 materialization used by the
pinned model:

```
https://storage.googleapis.com/flywire-data/codex/data/fafb/783/<file>
```

The bucket is publicly listable (Google Cloud Storage JSON API). The v783
directory is version-pinned by Codex — these are annotations **of the same
materialization** as `third_party/Drosophila_brain_model/Connectivity_783.parquet`.

Verification performed at acquisition time (`io_map_build.py` re-derives it):

- connectome unique neurons: **138,639**
- classification rows: 139,255 → **100.0 %** of connectome IDs carry an annotation
- consolidated types rows: 138,327 (99.59 % of them inside the connectome)
- author-deposited labels include the Bidaye-lab locomotor circuit IDs
  (MDN, BPN, Foxglove/Bluebell, BRK, DNp09/P9) — the same neurons used in
  the corresponding publications

Files:

| file | rows | key columns |
|---|---|---|
| classification.csv.gz | 139,255 | root_id, flow, super_class, class, sub_class, hemilineage, side, nerve |
| consolidated_cell_types.csv.gz | 138,327 | root_id, primary_type, additional_type(s) |
| neurons.csv.gz | 139,255 | root_id, group, nt_type, nt_type_score, per-NT scores |
| column_assignment.csv.gz | 45,528 | root_id, hemisphere, type, column_id, x, y, p, q (retinotopy) |
| labels.csv.gz | 4 M+ | root_id, label, user_name (author labels) |
| processed_labels.csv.gz | 700 k+ | root_id, processed_labels |
| names.csv.gz | 139 k+ | root_id, name, group |
| neuropil_synapse_table.csv.gz | ~4 M | per-neuron per-neuropil synapse counts |
| cell_stats.csv.gz | 139,246 | length/area/size |
| visual_neuron_types.csv.gz | 95,079 | type, family, subsystem, category, side |
| fw_and_hemibrain_types.csv.gz | 139,255 | cell_type ↔ hemibrain_type cross-match |
| classification_with_fw_and_hemibrain_types.csv.gz | 139,255 | classification + hemibrain match |
| coordinates.csv.gz | — | positions |
| connectivity_tags.csv.gz | — | connectivity-derived tags |
| neuropil_stats.csv | 78 | neuropil volumes/synapse stats |

License/attribution: FlyWire (Princeton/HHMI Janelia et al.) public data
release; use follows the FlyWire terms of service. The connectome itself
(v783) is bundled with the pinned Shiu et al. model (see `../THIRD_PARTY.md`).

Not modified after download; SHA-256 manifest:
`SHA256SUMS.txt` (generated locally at acquisition).
