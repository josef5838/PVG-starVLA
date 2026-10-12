### data generation
```bash
cd /export/home/haoliang/projects/starVLA
bash examples/Robotwin/generate_robotwin_data.sh --task rotate_qrcode --num 100 --gpu 0

```

### eval 3hist
```bash
CKPT=/export/home/haoliang/projects/starVLA/results/Checkpoints/ablation_vggt_omega_nostate_hist9_smoke/final_model/pytorch_model.pt
# RoboTwin_PVG does not ship the legacy fixed seed list. Provide one explicitly
# when reproducing an in-domain run.
SEEDS="${ROBOTWIN_SEEDS_FILE:?Set ROBOTWIN_SEEDS_FILE to a preserved seed.txt}"

cd /export/home/haoliang/projects/starVLA && \
DEPLOY_POLICY_TEMPLATE_PATH=/export/home/haoliang/projects/starVLA/examples/Robotwin/eval_files/deploy_policy_hist9.yml \
bash examples/Robotwin/eval_files/start_eval.sh \
  -m demo_clean \
  -n hist9_indomain \
  -c "${CKPT}" \
  --seeds-file "${SEEDS}" \
  rotate_qrcode \
  2>&1 | tee /tmp/hist9_eval.log

```

### continue training
```bash
cd /export/home/haoliang/projects/starVLA
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=1 /export/home/haoliang/miniconda3/envs/starVLA/bin/accelerate launch \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/ablation_vggt_omega_nostate.yaml \
  --run_id ablation_vggt_omega_nostate_hist9_smoke \
  --trainer.is_resume true \
  --trainer.max_train_steps 6000 \
  --trainer.num_warmup_steps 600 \
  --trainer.save_interval 2000 \
  --trainer.eval_interval 500 \
  2>&1 | tee /tmp/resume_train_hist9.log

```