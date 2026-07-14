#!/usr/bin/env bash
# Fleet monitor: polls job states until all targets reach a terminal state.
# Targets (by name): kappa-validation, ecg-adversarial-attack, base-e*, kappa-l*-e*,
# kappa-adaptive-e*. Exits when no target is in a non-terminal state.
# Terminal states: COMPLETED, FAILED, CANCELLED. Polls every POLL_SECS (default 180s).
set -uo pipefail
[ -f .env ] && set -a && . ./.env && set +a
NEBIUS="${NEBIUS:-$HOME/.nebius/bin/nebius}"
POLL_SECS="${POLL_SECS:-180}"
MAX_POLLS="${MAX_POLLS:-120}"     # ~6h cap

target_re='^(kappa-validation|ecg-adversarial-attack|base-e|kappa-l|kappa-adaptive)'

for i in $(seq 1 "$MAX_POLLS"); do
  json=$("$NEBIUS" ai job list --parent-id "$PARENT_ID" --format json 2>/dev/null)
  summary=$(printf '%s' "$json" | python3 -c "
import sys, json, re
d = json.load(sys.stdin)
tre = re.compile(r'$target_re')
TERM = {'COMPLETED','FAILED','CANCELLED'}
rows=[(i['metadata'].get('name',''), i.get('status',{}).get('state','?')) for i in d.get('items',[])]
tgt=[(n,s) for n,s in rows if tre.match(n)]
nonterm=[(n,s) for n,s in tgt if s not in TERM]
print('NONTERM', len(nonterm))
for n,s in sorted(tgt): print(f'  {n:<26} {s}')
")
  nonterm=$(printf '%s' "$summary" | awk '/^NONTERM/{print $2}')
  echo "[poll $i] $(date +%H:%M:%S)  não-terminais=$nonterm"
  printf '%s\n' "$summary" | grep -v '^NONTERM'
  if [ "${nonterm:-1}" = "0" ]; then
    echo "TODOS OS JOBS-ALVO TERMINARAM."
    exit 0
  fi
  sleep "$POLL_SECS"
done
echo "TETO DE POLLS ATINGIDO (ainda há jobs rodando)."
exit 2
