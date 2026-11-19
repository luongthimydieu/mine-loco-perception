# Roadmap

Prioritized by scientific value per GPU-hour.

## P0 — strengthen the core claim

1. **Scale-up seed replication.** Re-run `exp_scaleup_ablation.py` with 2 additional seeds
   (~8 h each). Report mean ± range for the four models. This is the cheapest way to answer
   “is +10.9 pp stable?”.
2. **Mixed-domain training.** Train `M_mix` on clean + physical dust (1:1). Test whether the
   clean-domain cost can be reduced below 10 pp without losing the dust gain.
3. **Real mine data.** 200–500 real underground track images (with permission). Evaluate
   zero-shot transfer and, if labels exist, fine-tune. This is the largest external-validity gain.

## P1 — deployment realism

4. **Multi-line / multi-echo LiDAR simulation.** The 6.5 m ceiling is a single-line geometry result;
   extend to 4/8/16 lines and quantify the new ceiling.
5. **Temporal video evaluation.** Current evaluation uses independent frames; add tracker-based
   metrics (MOTA/IDF1) on synthetic sequences.
6. **ONNX/TensorRT export and edge benchmark.** Current latency is PyTorch on an RTX 5060;
   measure Jetson/CPU and INT8.
7. **Per-image routing.** Estimate visibility from image contrast and route to M_clean/M_phys;
   compare with model soup and TTA.

## P2 — engineering

8. **Shared core library.** The system package and experiment scripts duplicate physics/fusion/LiDAR/
   health code; unify behind `mine_loco_system.core` with regression tests.
9. **Web demo.** A browser-side degradation + detection demo (ONNX Runtime Web) would make the
   project page interactive without a server.
10. **Documentation translations.** Keep English/Chinese README in sync; add a short video walkthrough.

## Known non-goals

- Building explosion-proof hardware.
- Claiming mine-field accuracy from synthetic dust.
- Replacing the corrected protocol with a leaderboard-style split.
