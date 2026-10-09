#!/usr/bin/env bash
# Execution window: keeps a grid from running 7h–18h Monday to Friday (working hours);
# outside that (nights and weekends) it runs normally. Every 30 s:
#   Mon–Fri and 7 ≤ hour < 18  -> SIGSTOP on the whole process group (launcher + xargs + runs), without losing state
#   otherwise                  -> SIGCONT
# Exits on its own when the group ends. Works for any grid launched with setsid
# (the PGID is the launcher's PID, and the children inherit the group).
# Usage: bash scripts/janela_execucao.sh <PGID> <name> [log]
set -u
PGID=${1:?launcher PGID}
NOME=${2:?grid name}
LOG=${3:-results/janela_execucao.log}
estado=rodando
while pgrep -g "$PGID" >/dev/null 2>&1; do
  h=$((10#$(date +%H)))
  dow=$(date +%u)  # 1 = Monday ... 7 = Sunday
  if [ "$dow" -le 5 ] && [ "$h" -ge 7 ] && [ "$h" -lt 18 ]; then
    if [ "$estado" = rodando ]; then
      kill -STOP -- "-$PGID" 2>/dev/null && estado=pausado && echo "$(date +%F\ %T) $NOME PAUSE (SIGSTOP group $PGID)" >> "$LOG"
    fi
  else
    if [ "$estado" = pausado ]; then
      kill -CONT -- "-$PGID" 2>/dev/null && estado=rodando && echo "$(date +%F\ %T) $NOME RESUME (SIGCONT group $PGID)" >> "$LOG"
    fi
  fi
  sleep 30
done
echo "$(date +%F\ %T) $NOME END (group $PGID finished)" >> "$LOG"
