#!/bin/bash
# Stress test: run LOCOMO benchmark N times and collect results
set -e

RUNS=${1:-5}
RESULTS_DIR="benchmark/results/stress"
mkdir -p "$RESULTS_DIR"

echo "=== LOCOMO Stress Test: $RUNS runs ==="
echo ""

for i in $(seq 1 $RUNS); do
    echo "--- Run $i/$RUNS ---"

    # Restart container for clean MCP state
    docker compose restart memforge > /dev/null 2>&1
    sleep 12

    # Verify health
    if ! curl -sf http://localhost:3100/health > /dev/null 2>&1; then
        echo "  ERROR: MemForge not healthy, skipping run $i"
        continue
    fi

    # Run benchmark
    PYTHONUNBUFFERED=1 RERANKER_ENABLED=true \
        python benchmark/locomo_eval.py --conversations 1 --skip-ingest --concurrency 3 2>&1 \
        | tee "$RESULTS_DIR/run_${i}.log" \
        | grep -E "^(OVERALL|Category|Single|Temporal|Multi|Open|Failure|MemForge)" || true

    # Copy results JSON
    cp benchmark/results/locomo_results.json "$RESULTS_DIR/run_${i}.json"

    echo ""
done

# Summary
echo "==========================================="
echo "STRESS TEST SUMMARY ($RUNS runs)"
echo "==========================================="
echo ""

python3 -c "
import json, sys, os
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
runs = []
for i in range(1, $RUNS + 1):
    p = results_dir / f'run_{i}.json'
    if p.exists():
        data = json.load(open(p))
        runs.append(data['summary'])

if not runs:
    print('No results found!')
    sys.exit(1)

cats = ['Single-hop', 'Temporal', 'Multi-hop', 'Open-domain']
print(f'{'Run':<6}', end='')
for c in cats:
    print(f'{c:>12}', end='')
print(f'{'OVERALL':>12}')
print('-' * 66)

overalls = []
cat_scores = {c: [] for c in cats}
for i, r in enumerate(runs, 1):
    print(f'Run {i:<3}', end='')
    for c in cats:
        score = r['by_category'].get(c, {}).get('llm_judge_pct', 0)
        cat_scores[c].append(score)
        print(f'{score:>11.1f}%', end='')
    overall = r['overall_llm_judge']
    overalls.append(overall)
    print(f'{overall:>11.1f}%')

print('-' * 66)
print(f'{'Mean':<6}', end='')
for c in cats:
    vals = cat_scores[c]
    mean = sum(vals)/len(vals)
    print(f'{mean:>11.1f}%', end='')
mean_overall = sum(overalls)/len(overalls)
print(f'{mean_overall:>11.1f}%')

print(f'{'Min':<6}', end='')
for c in cats:
    print(f'{min(cat_scores[c]):>11.1f}%', end='')
print(f'{min(overalls):>11.1f}%')

print(f'{'Max':<6}', end='')
for c in cats:
    print(f'{max(cat_scores[c]):>11.1f}%', end='')
print(f'{max(overalls):>11.1f}%')

spread = max(overalls) - min(overalls)
print(f'\nVariance: {spread:.1f}% spread (max-min)')
print(f'Mean overall: {mean_overall:.1f}% ± {spread/2:.1f}%')
print(f'MemMachine: 91.23%')
print(f'Consistently above MemMachine: {\"YES\" if min(overalls) > 91.23 else \"NO (min=\" + f\"{min(overalls):.1f}%\" + \")\"} ')
"
