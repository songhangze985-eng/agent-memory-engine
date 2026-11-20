#!/usr/bin/env bash
# Run from the repository root with three cached datasets and the frozen 99-item reference.
set -u
if [ "$#" -ne 5 ] && [ "$#" -ne 7 ]; then
    printf 'Usage: %s RESULTS_DIR LOCOMO_JSON LONGMEMEVAL_JSON MEMDATA_DIR MEMBENCH_REFERENCE [BASELINE_REPO BASELINE_RESULTS]\n' "$0" >&2
    exit 2
fi
results=$1
locomo=$2
longmem=$3
memdata=$4
reference=$5
baseline_repo=${6:-}
baseline_results=${7:-}
mkdir -p "$results"
python_bin=${PYTHON:-python3}
workers=${STATIC_WORKERS:-4}
export OPENBLAS_NUM_THREADS=1
failed=0
: > "$results/exit-codes.txt"
run() {
    local name=$1 code
    shift
    (
        "$python_bin" -c 'import shlex, sys; print("Command:", shlex.join(sys.argv[1:]), end="\n\n")' "$@"
        "$@"
        code=$?
        printf '\nExit code: %d\n' "$code"
        exit "$code"
    ) > "$results/$name.log" 2>&1
    code=$?
    printf '%s %d\n' "$name" "$code" | tee -a "$results/exit-codes.txt"
    if [ "$code" -ne 0 ]; then failed=1; fi
}

run 01-pytest "$python_bin" -m pytest -v
run 02-exact-ident "$python_bin" -m eval.exact_ident_eval --legacy --out "$results/02-exact-ident.json"
run 03-membench "$python_bin" -m eval.bipolar_ablation membench --offline --cohort-only --reference "$reference" --data-dir "$memdata" --out "$results/03-membench.json"
run 04-stress-precision "$python_bin" -m eval.stress_precision --out "$results/04-stress-precision.json"
run 05-locomo "$python_bin" -m eval.locomo.run --offline --sample 100 --workers "$workers" --data-dir "$locomo" --out "$results/05-locomo.json"
run 06-longmemeval "$python_bin" -m eval.longmemeval.run --offline --sample 100 --workers "$workers" --data-dir "$longmem" --out "$results/06-longmemeval.json"
run 07-bipolar-stress "$python_bin" -m eval.bipolar_ablation stress --offline --out "$results/07-bipolar-stress.json"
run 08-regression-guard "$python_bin" -m eval.regression_guard --frozen-only --results "$results"

run full-locomo "$python_bin" -m eval.locomo.run --all --workers "$workers" --data-dir "$locomo" --out "$results/full-locomo.json"
run full-long "$python_bin" -m eval.longmemeval.run --all --workers "$workers" --data-dir "$longmem" --out "$results/full-long.json"
run full-mem "$python_bin" -m eval.membench_static --workers "$workers" --data-dir "$memdata" --out "$results/full-mem.json"
run compact-locomo "$python_bin" -m eval.locomo.run --offline --sample 100 --consolidate --workers "$workers" --data-dir "$locomo" --out "$results/compact-locomo.json"
run compact-long "$python_bin" -m eval.longmemeval.run --offline --sample 100 --consolidate --workers "$workers" --data-dir "$longmem" --out "$results/compact-long.json"
run long-horizon "$python_bin" -m eval.long_horizon_static --out "$results/long-horizon.json"
run exact-static "$python_bin" -m eval.exact_ident_eval --out "$results/exact-static.json"
run engineering "$python_bin" -m eval.engineering_bench --out "$results/engineering.json"
run graph-10k "$python_bin" -m eval.engineering_bench --graph --out "$results/graph-10k.json"
run ablation "$python_bin" -m eval.ablation_matrix --locomo "$locomo" --longmem "$longmem" --memdata "$memdata" --workers "$workers" --out "$results/ablation.json"
run chrono "$python_bin" -m eval.chrono_eval --out "$results/chrono.json"
run version "$python_bin" -m eval.version_precision_eval --out "$results/version.json"
run chronicle-locomo "$python_bin" -m eval.chronicle_bench locomo --workers "$workers" --data-dir "$locomo" --out "$results/chronicle-locomo.json"
run chronicle-long "$python_bin" -m eval.chronicle_bench longmemeval --workers "$workers" --data-dir "$longmem" --out "$results/chronicle-long.json"
if [ -n "$baseline_repo" ]; then
    run answer-evidence "$python_bin" -m eval.evidence_ablation --baseline-repo "$baseline_repo" --out "$results/answer-evidence.json"
    run paired-10k "$python_bin" -m eval.engineering_bench --baseline-repo "$baseline_repo" --out "$results/paired-10k.json"
    run full-regression-guard "$python_bin" -m eval.regression_guard --results "$results" --baseline-results "$baseline_results"
else
    run full-regression-guard "$python_bin" -m eval.regression_guard --frozen-only --full-static --results "$results"
fi
run report "$python_bin" -m eval.static_report --results "$results"
exit "$failed"
