# Reference outputs

Small JSON summaries used by the docs and tests. They are not training data.

| File | Source |
|---|---|
| `simulation_metrics.json` | E1–E4 simulation metrics (original run) |
| `scaleup_metrics.json` | `real_metrics_scaleup.json` from the 5000-image ablation |
| `extension_eval.json` | V=30/15, TTA, model soup evaluation |
| `benchmark_inference.json` | batch-1 latency on RTX 5060 Laptop |
| `leak_diagnostic.json` | half-leaky vs independent evaluation |
| `per_class.json` | per-class AP@0.5 for four models × three sets |
| `threshold_sweep.json` | confidence/F1 sweep for M_clean and M_phys |

Regenerate with the scripts in `scripts/`; see `docs/reproduction.md`.
