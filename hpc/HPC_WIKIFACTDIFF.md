# CYCLEFT on IIT Delhi HPC — WikiFactDiff E1 Run Book (PBS/qsub)

Status: prepared 2026-11-23, updated for **PBS** (IIT Delhi HPC uses PBS
Pro).  **No PBS project ID, queue authorization or GPU allocation is
assumed or invented** — every such value is an explicit `<PLACEHOLDER>`
you must fill in from your HPC account/allocation letter.  Nothing in
this document has been executed on the HPC yet.

## 0. Division of work

| Task | Machine | Notes |
|---|---|---|
| Dataset download + preparation (manifest, event tables) | HPC login/CPU node | ~178 MB HF download + one 328k-row pass (~6 min on MacBook, faster on HPC) |
| Pilot training (64 train/event, pilot eval populations, 3 events) | HPC **GPU** node | MacBook GPU training is out of scope |
| Full training (500 train/event, complete held-out populations, 3 seeds) | HPC GPU node | after the pilot is reviewed |
| Analysis (metrics, matrices, plots) | either | CPU-only |

## 1. Environment setup (placeholders marked `<>`)

Development reference versions (from `hpc_requirements_key_versions.txt`):
python 3.11.16, torch 2.14.1, transformers 5.21.0, peft 0.21.2,
datasets 5.1.0, accelerate 1.15.0.  **Do not install unverified CUDA or
PyTorch versions** — prefer the HPC module stack; only pin exact versions
the cluster actually provides.

```bash
# example — adjust module names to what `module avail` actually shows:
# module load python/3.11 cuda/12.1
python -m venv .venv-hpc
source .venv-hpc/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match the CUDA module
pip install transformers peft datasets accelerate pyyaml
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Sync from GitHub

```bash
git clone git@github.com:himanshusuryawanshi/CYCLEFT.git
cd CYCLEFT
git checkout fix/wikifactdiff-hpc-readiness   # or the merged branch name
git log --oneline -3
```

HF dataset: `Orange/WikiFactDiff` config `20210104-20230227_legacy`
(revision pinned `bb17ffbff7b2d28e4cd12e251af3db50d7fa18ea`, ~178 MB).
If HF is blocked, download the parquet on the login node or transfer
`~/.cache/huggingface/datasets` from the MacBook.

## 3. Dataset preparation (login/CPU node)

```bash
python scripts/e1_prepare.py --config configs/e1_stream_wikifactdiff.yaml \
    --mode full --out data/e1/wikifactdiff_manifest_full.json
```

Acceptance: the manifest's `counts` must show 327,688 rows = 327,688
unique facts; decision totals new 410,449 / obsolete 39,384 / static
4,532; partitions summing to 327,688 facts.  Commit the manifest.

## 4. PBS `qsub` job template

Save as `hpc/train_e1_wfd.pbs`.  **Every `<ANGLE_BRACKET>` value must be
replaced with a value from your HPC documentation/allocation.  Do not
guess.**

```bash
#!/bin/bash
#PBS -N cycleft-e1-wfd
#PBS -P <PROJECT_ID>                # your IIT Delhi PBS project/queue code
#PBS -q <GPU_QUEUE>                 # authorized GPU queue — confirm with HPC docs
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB
#PBS -l walltime=04:00:00
#PBS -o logs/${PBS_JOBNAME}-${PBS_JOBID}.out
#PBS -e logs/${PBS_JOBNAME}-${PBS_JOBID}.err

set -euo pipefail                   # fail hard on any error (no || true)
cd "$PBS_O_WORKDIR"

# --- environment -----------------------------------------------------
source .venv-hpc/bin/activate
unset PYTORCH_ENABLE_MPS_FALLBACK   # must remain unset: no silent CPU fallback

# --- CUDA preflight (fail before training, never mid-run) ------------
python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA not available — check module/queue"
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA runtime:", torch.version.cuda)
print("Torch:", torch.__version__)
free, total = torch.cuda.mem_get_info()
print(f"Memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
PYEOF

# --- dataset preparation (idempotent; skipped if manifest exists) ----
if [ ! -f data/e1/wikifactdiff_manifest_full.json ]; then
    python scripts/e1_prepare.py \
        --config configs/e1_stream_wikifactdiff.yaml \
        --mode full \
        --out data/e1/wikifactdiff_manifest_full.json
fi

# --- training (single seed; repeat per seed with --seed/--run-id) ----
SEED="${SEED:-42}"
python scripts/e1_run.py \
    --config configs/e1_stream_wikifactdiff.yaml \
    --mode "${MODE:-pilot}" \
    --seed "$SEED" \
    --run-id "wfd_${MODE:-pilot}_s${SEED}_$(date +%Y%m%d_%H%M)" \
    --device cuda

echo "JOB COMPLETE: $PBS_JOBID"
```

Submission and monitoring:

```bash
mkdir -p logs
qsub hpc/train_e1_wfd.pbs           # prints the job id, e.g. 12345.<server>
qstat -u "$USER"                    # job status
qstat -f <JOB_ID>                   # detailed status
tail -f logs/cycleft-e1-wfd-<JOB_ID>.out
# delete a stuck job (only if you are sure):
# qdel <JOB_ID>
```

Failure handling: with `set -euo pipefail` the job stops at the first
failing command and the `.err` log carries the traceback.  A failed job
leaves `results/e1/<run_id>/` and `checkpoints/e1/<run_id>/` in place —
inspect them; resume with `--start-event {2,3}` only if the earlier
checkpoints exist (the pipeline aborts with an explicit error otherwise;
it never silently retrains).

## 5. Seeds and runs

| Run | Command line | Output dir |
|---|---|---|
| Pilot | `MODE=pilot qsub ...` (SEED=42 default) | `results/e1/wfd_pilot_s42_*` |
| Full s42 | `MODE=full SEED=42 qsub ...` | `results/e1/wfd_full_s42_*` |
| Full s43 | `MODE=full SEED=43 qsub ...` | `results/e1/wfd_full_s43_*` |
| Full s44 | `MODE=full SEED=44 qsub ...` | `results/e1/wfd_full_s44_*` |

Run seeds **sequentially** (one job at a time) unless your allocation
explicitly allows parallel GPU jobs.

## 6. Result synchronization back to GitHub/MacBook

```bash
git add data/e1/*.json results/e1/wfd_*/   # JSON artifacts + summaries
git commit -m "E1 WikiFactDiff results: <run ids>"
git push origin fix/wikifactdiff-hpc-readiness
# large prediction/checkpoint files: tar + transfer separately if policy
# disallows them in git
tar czf wfd_results_$(date +%Y%m%d).tar.gz results/e1/wfd_* checkpoints/e1/wfd_*
```
