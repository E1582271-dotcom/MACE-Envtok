#!/usr/bin/env bash
# One-screen summary of a training run.
#   bash scripts/nscc/status.sh [run_dir] [job_id]
#   (defaults: $ENVTOK_SCRATCH/runs/envtok_8gpu and the newest envtok-train job)
cd "$(dirname "$0")/../.."
. scripts/nscc/env.sh >/dev/null 2>&1
RUN=${1:-$ENVTOK_SCRATCH/runs/envtok_8gpu}
JOB=${2:-$(qstat -u "$USER" 2>/dev/null | awk '/envtok-tr/ && $10 == "R" {print $1}' | head -1)}
JOB=${JOB%%.*}

echo "== jobs"
qstat -u "$USER" 2>/dev/null | tail -n +6

python - "$RUN" "${JOB:-}" <<'PY'
import csv, json, os, sys, time
run, job = sys.argv[1], sys.argv[2]
def rows(name):
    path = os.path.join(run, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []
cfg = json.load(open(os.path.join(run, "config.json"))) if os.path.exists(os.path.join(run, "config.json")) else {}
epochs = [r for r in rows("epochs.csv") if r.get("epoch") not in (None, "", "-1")]
steps = rows("steps.csv")
train = cfg.get("train", {})
world = cfg.get("world_size", "?")
print(f"\n== run {run}")
print(f"GPUs {world}, global batch {cfg.get('global_batch')}, steps/epoch {cfg.get('steps_per_epoch')}, "
      f"train {cfg.get('n_train')}, val {cfg.get('n_val')}, max epochs {train.get('max_epochs')}, patience {train.get('patience')}")
if os.path.exists(os.path.join(run, "FINISHED")):
    print("FINISHED:", open(os.path.join(run, "FINISHED")).read().strip())
if steps:
    s = steps[-1]
    print(f"\nlatest step: epoch {s['epoch']} step {s['step']} (global {s['global_step']}), lr {float(s['lr']):.3g}, "
          f"loss {float(s['loss']):.5g}, {float(s['structures_per_s']):.0f} structures/s, "
          f"{float(s['seconds_per_step']):.3f} s/step, rank time max/mean {float(s['rank_time_max_over_mean']):.3f}, "
          f"GPU mem {float(s['max_mem_GB']):.1f} GB")
if epochs:
    keys = ["loss", "energy_mae_meV_per_atom", "force_mae_meV_per_A", "stress_mae_meV_per_A3"]
    print("\nlast epochs (val, EMA weights):")
    print(f"  {'epoch':>5} {'lr':>9} {'loss':>11} {'E meV/at':>9} {'F meV/A':>8} {'S meV/A3':>9} {'sec':>6} {'bad':>4}")
    for r in epochs[-5:]:
        print(f"  {r['epoch']:>5} {float(r['lr']):9.3g} {float(r['loss']):11.5g} {float(r['energy_mae_meV_per_atom']):9.2f} "
              f"{float(r['force_mae_meV_per_A']):8.1f} {float(r['stress_mae_meV_per_A3']):9.2f} "
              f"{float(r.get('epoch_seconds') or 0):6.0f} {r.get('bad_epochs', ''):>4}")
    best = min(epochs, key=lambda r: float(r["loss"]))
    print(f"best so far: epoch {best['epoch']} (loss {float(best['loss']):.5g}, "
          f"F {float(best['force_mae_meV_per_A']):.1f} meV/A, E {float(best['energy_mae_meV_per_atom']):.2f} meV/atom)")
    secs = [float(r["epoch_seconds"]) for r in epochs[-20:] if r.get("epoch_seconds")]
    if secs:
        per = sum(secs) / len(secs)
        done = int(epochs[-1]["epoch"]) + 1
        left_max = int(train.get("max_epochs", 1000)) - done
        bad = int(epochs[-1].get("bad_epochs") or 0)
        left_min = max(0, int(train.get("patience", 40)) - bad)
        print(f"epoch time {per:.0f} s; remaining: at least {left_min} epochs ({left_min * per / 3600:.1f} h, if no further "
              f"improvement), at most {left_max} epochs ({left_max * per / 3600:.1f} h)")
        gpus = world if isinstance(world, int) else 0
        print(f"GPU-hours in epochs so far: {sum(float(r.get('epoch_seconds') or 0) for r in epochs) * gpus / 3600:.1f}")
PY

if [ -n "${JOB:-}" ]; then
    logs=$(ls "$ENVTOK_SCRATCH"/runs/logs/"$JOB".*.log 2>/dev/null)
    if [ -n "$logs" ]; then
        echo; echo "== problems in per-node logs of job $JOB"
        grep -h -E "Traceback|Error|error:|non-finite|exited with code" $logs | grep -v -i "warn" | sort | uniq -c | head -10 || true
        [ -z "$(grep -l -E 'Traceback|Error' $logs 2>/dev/null)" ] && echo "none"
    fi
fi
