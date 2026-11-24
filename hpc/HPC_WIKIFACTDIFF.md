# CYCLEFT on IIT Delhi HPC — WikiFactDiff E1 Run Book

Status: prepared 2026-11-23. **No PBS/Slurm project identifier is assumed
and no GPU allocation is presumed** — fill in the marked placeholders with
the values your HPC account provides. Nothing in this document has been
executed on the HPC yet.

## 0. What runs where

| Task | Machine | Why |
|---|---|---|
| Dataset download + preparation (manifest, event tables) | HPC login/CPU node | 178 MB HF download + one 328k-row pass (~6 min on MacBook; faster on HPC) |
| Pilot training (64/event, 100 eval/event, 3 events) | HPC **GPU** node | MacBook GPU training is out of scope per project policy |
| Full training (500/event, full held-out eval, 3 seeds) | HPC GPU node | scale |
| Analysis (metrics, matrices, plots) | either | CPU-only |

## 1. Environment setup (placeholders marked `<>`)

The development environment (recorded in `hpc_requirements_key_versions.txt`)
is: Python 3.11.16, torch 2.x CUDA build, transformers, peft, datasets,
accelerate. **Do not install unverified CUDA/PyTorch versions** — prefer the
HPC module system if one exists:

```bash
# example (adjust to the HPC's actual module names):
# module load python/3.11 cuda/12.1
python -m venv .venv-hpc
source .venv-hpc/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match the module's CUDA
pip install transformers peft datasets accelerate pyyaml
pip install -r hpc_requirements_key_versions.txt --dry-run && pip install -r hpc_requirements_key_versions.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Record the resolved versions into the run manifest (`runtime_metadata.json`
is written automatically by the pipeline).

## 2. Sync from GitHub

```bash
git clone git@github.com:himanshusuryawanshi/CYCLEFT.git
cd CYCLEFT
git checkout feature/wikifactdiff-e1     # branch with the WikiFactDiff adapter
git log --oneline -2                     # expect 99a48ce feat(e1) ... on 9ff5f7e feat(data)
```

HF dataset note: `Orange/WikiFactDiff` config `20210104-20230227_legacy`
(revision pinned `bb17ffbff7b2d28e4cd12e251af3db50d7fa18ea`, ~178 MB parquet).
If the HPC blocks HF downloads, download the parquet on the login node or
transfer `~/.cache/huggingface/datasets` from the MacBook.

## 3. Dataset preparation job (CPU)

```bash
python scripts/e1_prepare.py --config configs/e1_stream_wikifactdiff.yaml \
    --mode full --out data/e1/wikifactdiff_manifest_full.json
```

Expected verified counts (already reproduced locally): 327,688 rows =
327,688 unique facts; expanded 328,079 examples (one per new object);
decision totals new 410,449 / obsolete 39,384 / static 4,532; partitions
112,001/13,970/13,840 subjects. The manifest is the acceptance artifact.

## 4. GPU training job (template)

```bash
# hpc/train_e1_wfd.sh — adapt scheduler keywords to the HPC (Slurm shown;
#  delete/replace #PBS lines if the cluster uses PBS).
#SBATCH --job-name=cycleft-e1-wfd
#SBATCH --account=<YOUR_IITD_PROJECT>          # <-- fill in; do not guess
#SBATCH --partition=<GPU_PARTITION>            # <-- fill in
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
source .venv-hpc/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=            # must remain unset: no silent CPU fallback
python scripts/e1_run.py --config configs/e1_stream_wikifactdiff.yaml \
    --mode pilot --run-id wfd_pilot_$(date +%Y%m%d_%H%M) \
    --device cuda || true
# NOTE: scripts/e1_run.py currently accepts --device {mps,cpu}; for CUDA add
# 'cuda' to its choices (one-line change) OR set device: null in the config
# and run on a GPU-visible node so torch auto-selects CUDA.
```

Sequence: pilot first (64 train / 100 eval per event), inspect
`results/e1/<run_id>/` (dataset_manifest.json, baseline_metrics.json,
event_*_eval.json, event_metrics.json), then run `--mode full` for the
three seeds:

```bash
for seed in 42 43 44; do
  python scripts/e1_run.py --config configs/e1_stream_wikifactdiff.yaml \
      --mode full --seed $seed --run-id wfd_full_s${seed}
done
```

## 5. Checkpoints, manifests, results

| Artifact | Location | Notes |
|---|---|---|
| Adapter checkpoints + tokenizer + provenance | `checkpoints/e1/<run_id>/event_<t>/` | per-event; continuing adapter lineage |
| Dataset manifest (partition, salt, algorithm version, exclusions) | `data/e1/wikifactdiff_manifest*.json` | acceptance artifact; commit it |
| Metrics matrices | `results/e1/<run_id>/event_metrics.json` | EM/F1/ppl state × test-event |
| Predictions | `results/e1/<run_id>/predictions.jsonl` | incl. `test_event`, `valid_answers` |
| Runtime versions | `results/e1/<run_id>/runtime_metadata.json` | torch/CUDA/GPU name |

## 6. Result collection & recovery

- `git add data/e1/*.json results/e1/<run_id>/*.json` and push (do not push
  large prediction files if policy disallows; tar them instead).
- Recovery: `scripts/e1_run.py --start-event {2,3} --run-id <id>` resumes
  from the saved per-event checkpoints; a missing earlier checkpoint or
  baseline aborts with an explicit error (never silently retrains).
- A crashed GPU job does NOT corrupt prior events: each event loads the
  base model fresh and continues from the previous checkpoint directory.
- Never delete `checkpoints/e1/<run_id>/event_*` — they are the resume and
  audit anchors.

## 7. Post-run metric sweep (M1/M3 populations)

The M1 (trained-fact paraphrases) and M3 (neighborhood probes, persistence
probes) populations are written by the adapter into the dataset manifest
(`sweep_rows`). Evaluate them per checkpoint with the multi-reference
evaluator (checkpoint per event + base model) and aggregate with
`cycleft.e1.wfd_metrics` (`m1_update_efficacy`, `m2_heldout_transfer`,
`m3_neighborhood_retention`, `m3_persistence_rate`). A dedicated sweep
driver can be added on request; the populations and functions are ready.
