# SurfloDiT checkpoint handoff

Create one bundle per checkpoint on the training machine. The packer reads
`freeze_backbone` from the saved run config. Frozen Surflo weights are omitted;
trained Surflo weights remain in the policy checkpoint. Bundles contain the
policy checkpoint, normalization statistics, portable config, a SHA-256
manifest, and the exact StarVLA inference source used to build the bundle.
They never contain the released Surflo checkpoint or Surflo source code.

```bash
cd /path/to/PVG-starVLA
python tools/package_surflo_checkpoint.py \
  --checkpoint results/Checkpoints/surflo_camera3_state_token_20k/checkpoints/steps_20000_pytorch_model.pt \
  --output results/TransferBundles/surflo_camera3_state_token_20k
```

Copy the complete bundle directory to RTX with the available transfer medium,
for example `rsync -a --info=progress2`. Do not modify bundle files after
packaging. On RTX, install the matching Surflo source and, for frozen bundles,
the released checkpoint and model config. The verifier checks their SHA-256
hashes against the manifest; the RTX Surflo path may differ from the training
machine path.

```bash
BUNDLE=/path/to/surflo_camera3_state_token_20k
SURFLO_ROOT=/path/to/Surflo
PYTHONPATH="$BUNDLE/code" python "$BUNDLE/code/tools/verify_surflo_bundle.py" \
  --bundle "$BUNDLE" --surflo-root "$SURFLO_ROOT"
```

The verifier checks every bundle file, the local Surflo Python source, the
released checkpoint and model config when referenced, then strictly restores
the policy with the bundled architecture code. For a trained-backbone bundle,
the model config is bundled and no release checkpoint is needed.

Start the policy server from the same code snapshot:

```bash
PYTHONPATH="$BUNDLE/code" python \
  "$BUNDLE/code/examples/simBenchmarks/Behavior/serve_vggt_omega_2026.py" \
  --bundle "$BUNDLE" --surflo-root "$SURFLO_ROOT" \
  --tasks-jsonl /path/to/behavior-1k-2026/meta/tasks.jsonl \
  --host 127.0.0.1 --port 8000 --execute-horizon 4 --image-size 518
```

This procedure checks code and weight compatibility. It does not establish
closed-loop task success; run the BEHAVIOR evaluator after the server loads.
